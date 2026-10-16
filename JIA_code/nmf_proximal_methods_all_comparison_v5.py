from pathlib import Path
# -*- coding: utf-8 -*-
"""
Orthogonal NMF via Proximal Gradient Methods
=============================================

Fits a sparsity-regularized, (semi-)orthogonal Nonnegative Matrix
Factorisation

    min_{W>=0, H>=0}  ||V - W H||_F^2  +  mu * ||W^T W - I||_F^2  + sparsity(H)

on TF-IDF features from the 20-newsgroups corpus, and compares four
proximal-gradient-style solvers for the (W, H) subproblem:

    PSD     Proximal Steepest Descent            (baseline: no scaling)
    PCG     Proximal (Fletcher-Reeves) CG         (conjugate directions)
    SPM     Spectral Proximal Method              (REFERENCE implementation)
    PBFGS   Proximal Diagonal-BFGS                (diagonal quasi-Newton)

All four methods use the same interface, stopping rule, line search,
warm-start strategy, diagnostics and evaluation metrics so that their
results are directly comparable. Only each method's *search direction* and
*curvature/scaling update* differ. The experiment pipeline can run PSD,
PCG, SPM and PBFGS together from the same warm start and generate unified
tables and comparison figures.

Typical usage
-------------
    python nmf_proximal_methods.py <max_itr> <MAXITER> <init_mode> \
                                    <update_rule> <opt_method> <n_components>

    max_itr      : # of multiplicative-update NMF iterations used to warm-start
                   W, H before the optimizer runs (0 = skip, use raw init)
    MAXITER      : # of proximal-gradient iterations (0 = skip optimizer)
    init_mode    : 0 = random init, 1 = NNDSVD init
    update_rule  : 0 = Euclidean-distance NMF warm start, 1 = divergence (KL) warm start
    opt_method   : retained for backward compatibility; all four methods are run
    n_components : number of topics (rank of the factorisation)

Author: yihan
"""

import re
import sys
import cProfile
import pstats
from time import time

import numpy as np
import numpy.linalg as LA
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


# =====================================================================
# Global configuration
# =====================================================================
N_SAMPLES = 2000
N_FEATURES = 1000
N_TOP_WORDS = 20

TOL = 1e-6                     # shared stopping tolerance
MU = 10.0                      # orthogonality-penalty weight (was 0.5: too small to matter)
LAMBDA_SPARSE = 1e-3           # L1-sparsity weight applied to H (was 1e-8: effectively zero)
# For an L1 term lambda*||H||_1, the proximal threshold is alpha*lambda
# (or alpha*lambda*D for a diagonal variable metric D), not sqrt(2*lambda).

EPS = 1e-12                    # generic safe-division guard
# Fixed, iteration-independent bounds on the spectral inverse-metric entries b:
#     B_MIN <= b_ij <= B_MAX   (uniformly positive-definite diagonal metric,
# the standing assumption of variable-metric proximal-gradient convergence
# theory). Same values as PBFGS's scale_bounds so the two are comparable.
B_MIN = 1e-8
B_MAX = 1e8

# ---------------------------------------------------------------------
# Hyperparameter glossary (paper notation -> code constant)
# ---------------------------------------------------------------------
#   gamma (orthogonality weight)  -> MU             = 10.0
#   beta  (sparsity weight)       -> LAMBDA_SPARSE   = 1e-3   (L1 proximal threshold = alpha*lambda)
#   tau   (convergence tolerance) -> TOL             = 1e-6   (relative step-change criterion)
#   Armijo sufficient-decrease    -> ARMIJO_C1       = 1e-4
#   Armijo backtracking factor    -> ARMIJO_BETA     = 0.5
#   PBFGS diagonal damping        -> PBFGS_DAMPING   = 0.2
# MU / LAMBDA_SPARSE: run `sensitivity_study()` on your data to choose them;
# do not report them as grid-search-selected unless you have run that sweep.
ARMIJO_C1 = 1e-4
ARMIJO_BETA = 0.5
PBFGS_DAMPING = 0.2

# Benchmark controls
COMMON_START = True          # all 8 methods start from exactly the same W,H
PROFILE_OPTIMIZERS = False   # cProfile is expensive; enable only for profiling
BASELINE_L1 = 1e-3           # L1 strength for SparseNMF baseline
# Iteration protocol (python script.py max_itr MAXITER ...):
#   baselines (NMF-ED, NMF-KL, ONMF, SparseNMF): raw init -> MAXITER iterations
#   proximal (PSD, SPM, PBFGS, PCG): raw init -> max_itr NMF-ED iterations
#                                    -> MAXITER proximal iterations from that result
# Set True only if you want baselines to get max_itr + MAXITER iterations instead.
EQUAL_ITERATION_BUDGET = False

METHOD_NAMES = {0: "PSD", 1: "PCG", 2: "SPM", 3: "PBFGS"}
ALL_METHODS = ["PSD", "SPM", "PBFGS", "PCG", "NMF-ED", "NMF-KL", "ONMF", "SparseNMF"]
UPDATE_RULE_NAMES = {0: "EuclideanDistance", 1: "Divergence"}
BASELINE_NAMES = {"NMF-ED": "vanilla NMF (Euclidean)", "NMF-KL": "vanilla NMF (KL divergence)",
                   "ONMF": "orthogonal NMF (Ding et al. 2006)", "SparseNMF": "sparse NMF (Hoyer-style, L1 on H)"}


# =====================================================================
# 1. Data loading & preprocessing
# =====================================================================
def load_and_preprocess(n_samples):
    """Load a slice of 20-newsgroups and clean it: strip punctuation,
    lowercase, remove stopwords, stem. Returns a DataFrame with the
    cleaned text plus a before/after sentence count for sanity checking.
    """
    print("Loading dataset...")
    t0 = time()
    raw = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
    )
    docs = raw.data[n_samples:n_samples * 2]
    df = pd.DataFrame(docs, columns=["text"])
    df["num_sent_before"] = df["text"].apply(sent_tokenize).apply(len)

    stop_words = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    def clean(text):
        text = re.sub(r"[^\w\s]", "", text).lower()
        tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    df["text"] = df["text"].apply(clean)
    df["num_sent_after"] = df["text"].apply(sent_tokenize).apply(len)

    print("done in %0.3fs." % (time() - t0))
    return df


def extract_tfidf(texts, max_features):
    vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=max_features, stop_words="english"
    )
    print("Extracting tf-idf features...")
    t0 = time()
    X = vectorizer.fit_transform(texts)
    print("done in %0.3fs." % (time() - t0))
    return X, vectorizer


# =====================================================================
# 2. Initialization
# =====================================================================
def to_dense_array(V):
    """Return V as a plain ndarray, never a numpy.matrix.

    scipy sparse `.todense()` returns `numpy.matrix`, whose `*` operator
    means matrix multiplication rather than an elementwise product. If
    that type ever leaks into a gradient (e.g. via `V - W@H` when V is
    sparse), every later elementwise op in the proximal-gradient solvers
    silently turns into a matrix product and crashes with a shape error.
    `.toarray()` (sparse) and `np.asarray()` (anything else, including
    numpy.matrix) both strip this down to a base ndarray.
    """
    if hasattr(V, "toarray"):
        return V.toarray()
    return np.asarray(V)


def random_initialization(V, n_components):
    """Uses the caller's global NumPy random state (set via
    `np.random.seed(...)`) rather than a fixed internal seed, so that
    repeated-seed experiments (see `run_experiment`) actually vary the
    initialization across seeds."""
    W = np.random.uniform(0, 1, (V.shape[0], n_components))
    H = np.random.uniform(0, 1, (n_components, V.shape[1]))
    return W, H


def nndsvd_initialization(V, n_components):
    """Nonnegative Double SVD initialization (Boutsidis & Gallopoulos)."""
    u, s, vt = np.linalg.svd(to_dense_array(V), full_matrices=False)
    v = vt.T
    W = np.zeros((V.shape[0], n_components))
    H = np.zeros((n_components, V.shape[1]))

    W[:, 0] = np.sqrt(s[0]) * np.abs(np.asarray(u[:, 0]).flatten())
    H[0, :] = np.sqrt(s[0]) * np.abs(np.asarray(v[:, 0]).flatten())

    for i in range(1, n_components):
        ui, vi = np.asarray(u[:, i]).flatten(), np.asarray(v[:, i]).flatten()
        ui_pos, ui_neg = np.clip(ui, 0, None), np.clip(-ui, 0, None)
        vi_pos, vi_neg = np.clip(vi, 0, None), np.clip(-vi, 0, None)

        ui_pos_n, ui_neg_n = np.linalg.norm(ui_pos), np.linalg.norm(ui_neg)
        vi_pos_n, vi_neg_n = np.linalg.norm(vi_pos), np.linalg.norm(vi_neg)

        norm_pos, norm_neg = ui_pos_n * vi_pos_n, ui_neg_n * vi_neg_n

        if norm_pos >= norm_neg:
            W[:, i] = np.sqrt(s[i] * norm_pos) / max(ui_pos_n, EPS) * ui_pos
            H[i, :] = np.sqrt(s[i] * norm_pos) / max(vi_pos_n, EPS) * vi_pos
        else:
            W[:, i] = np.sqrt(s[i] * norm_neg) / max(ui_neg_n, EPS) * ui_neg
            H[i, :] = np.sqrt(s[i] * norm_neg) / max(vi_neg_n, EPS) * vi_neg

    return W, H


def initialize(V, n_components, init_mode):
    if init_mode == 0:
        return random_initialization(V, n_components)
    if init_mode == 1:
        return nndsvd_initialization(V, n_components)
    raise ValueError(f"Unknown init_mode: {init_mode}")


# =====================================================================
# 3. Multiplicative-update NMF baselines (warm start)
# =====================================================================
def ed_method(V, n_components, max_iter, init_mode, init=None):
    """Lee-Seung multiplicative updates minimizing ||V - WH||_F."""
    if init is None:
        W, H = initialize(V, n_components, init_mode)
    else:                                   # start exactly where the caller says
        W, H = init[0].copy(), init[1].copy()
    eps = 1e-10
    history = []
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
        history.append(np.linalg.norm(V - W @ H, "fro") ** 2)

    print("(Euclidean Distance NMF)")
    print("Final factorisation loss =", history[-1] if history else None, "\n")
    return W, H, history


def dvg_method(V, n_components, max_iter, init_mode, init=None):
    """Lee-Seung multiplicative updates minimizing generalized KL divergence."""
    if init is None:
        W, H = initialize(V, n_components, init_mode)
    else:                                   # start exactly where the caller says
        W, H = init[0].copy(), init[1].copy()
    V = to_dense_array(V)
    eps = 1e-10
    history = []
    for _ in range(max_iter):
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.sum(axis=0)[:, None] + eps)
        WH = W @ H + eps
        W *= ((V / WH) @ H.T) / (H.sum(axis=1)[None, :] + eps)
        WH = W @ H + eps
        history.append(np.linalg.norm(V - W @ H, "fro") ** 2)

    print("(Divergence NMF)")
    print("Final factorisation loss =", history[-1] if history else None, "\n")
    return W, H, history


def onmf_method(V, n_components, max_iter, init_mode):
    """Orthogonal NMF baseline (Ding, Li, Peng & Park, 2006): multiplicative
    updates that drive W toward column-orthogonality without an explicit
    penalty term:
        H <- H * (W^T V) / (W^T W H)
        W <- W * (V H^T) / (W (W^T (V H^T)))
    """
    W, H = initialize(V, n_components, init_mode)
    eps = 1e-10
    history = []
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        VHt = V @ H.T
        W *= np.sqrt(VHt / (W @ (W.T @ VHt) + eps))  # Ding et al. 2006 needs the sqrt
        history.append(np.linalg.norm(V - W @ H, "fro") ** 2)

    print("(Orthogonal NMF - Ding et al. 2006)")
    print("Final factorisation loss =", history[-1] if history else None,
          "| ||W^T W - I||_F =", f_orthogonality(W) ** 0.5, "\n")
    return W, H, history


def sparse_nmf_method(V, n_components, max_iter, init_mode, l1_h=1e-3):
    """Sparse NMF baseline (Hoyer, 2004 style): L1 penalty on H folded into
    the multiplicative denominator, with W's columns renormalized to unit
    norm each iteration (mass absorbed into H) so the penalty is
    well-defined and comparable across iterations.
    """
    W, H = initialize(V, n_components, init_mode)
    eps = 1e-10
    history = []
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + l1_h + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)
        norms = np.linalg.norm(W, axis=0, keepdims=True) + eps
        W /= norms
        H *= norms.T
        history.append(np.linalg.norm(V - W @ H, "fro") ** 2)

    print(f"(Sparse NMF, l1_h={l1_h})")
    print("Final factorisation loss =", history[-1] if history else None, "\n")
    return W, H, history



def _baseline_from_start(name, V, W0, H0, max_iter, l1_h=BASELINE_L1):
    """Run a baseline from a supplied common starting point.

    This avoids the previous hidden bias where each baseline generated its
    own initialization while the proximal methods started from an NMF
    warm-start. All methods in the benchmark now receive the same W0,H0.
    """
    W, H = W0.copy(), H0.copy()
    V = to_dense_array(V)
    eps = 1e-10
    history = []

    for _ in range(max_iter):
        if name == "NMF-ED":
            H *= (W.T @ V) / (W.T @ W @ H + eps)
            W *= (V @ H.T) / (W @ H @ H.T + eps)

        elif name == "NMF-KL":
            WH = W @ H + eps
            H *= (W.T @ (V / WH)) / (W.sum(axis=0)[:, None] + eps)
            WH = W @ H + eps
            W *= ((V / WH) @ H.T) / (H.sum(axis=1)[None, :] + eps)

        elif name == "ONMF":
            # W-orthogonal MU rule used in the existing implementation:
            # W <- W .* (V H^T) ./ (W W^T V H^T)
            H *= (W.T @ V) / (W.T @ W @ H + eps)
            VHt = V @ H.T
            W *= np.sqrt(VHt / (W @ (W.T @ VHt) + eps))  # Ding et al. 2006 needs the sqrt

        elif name == "SparseNMF":
            H *= (W.T @ V) / (W.T @ W @ H + l1_h + eps)
            W *= (V @ H.T) / (W @ H @ H.T + eps)
            norms = np.linalg.norm(W, axis=0, keepdims=True) + eps
            W /= norms
            H *= norms.T

        else:
            raise ValueError(f"Unknown baseline: {name}")

        history.append(f_factorisation(V, W, H))

    return W, H, history


BASELINE_METHODS = {
    "NMF-ED": ed_method,
    "NMF-KL": dvg_method,
    "ONMF": onmf_method,
    "SparseNMF": sparse_nmf_method,
}


# =====================================================================
# 4. Shared objective, gradients, and proximal/scaling operators
# =====================================================================
def f_factorisation(V, W, H):
    return np.linalg.norm(V - W @ H, "fro") ** 2


def f_orthogonality(W):
    r = W.shape[1]
    return np.linalg.norm(W.T @ W - np.identity(r), "fro") ** 2


def f_W(V, W, H, mu=None):
    mu = MU if mu is None else mu
    return f_factorisation(V, W, H) + mu * f_orthogonality(W)


def f_H(V, W, H, mu=None):
    mu = MU if mu is None else mu
    # H does not enter the orthogonality penalty; kept for interface symmetry
    # with f_W so both share the same call signature in the line search.
    return f_factorisation(V, W, H) + mu * f_orthogonality(W)


def df_W(V, W, H, mu=None):
    mu = MU if mu is None else mu
    r = W.shape[1]
    return -2 * (V - W @ H) @ H.T + mu * 4 * W @ (W.T @ W - np.identity(r))


def df_H(V, W, H):
    return -2 * W.T @ (V - W @ H)


def prox_l1_nonnegative(Z, threshold):
    """Proximal map of threshold*||H||_1 plus H>=0.

    `threshold` may be a scalar (identity metric) or an array for a
    diagonal variable metric.
    """
    return np.maximum(np.asarray(Z) - threshold, 0.0)


_BOUND_STATS = {"entries": 0, "clipped_low": 0, "clipped_high": 0}


def spectral_metric(s, y, eps=EPS, b_min=B_MIN, b_max=B_MAX):
    """Elementwise Barzilai-Borwein-type diagonal scaling from a secant pair,
    safeguarded so that b_min <= b <= b_max for every entry."""
    s = np.asarray(s)
    omega = (np.trace(s.T @ s) - np.trace(s.T @ np.asarray(y))) / (np.sum(s ** 4) + eps)
    if omega < 0:
        b = 1.0 / ((np.trace(s.T @ np.asarray(y)) / (np.trace(s.T @ s) + eps)) * np.ones_like(s))
    else:
        b = 1.0 / (1 + omega * (s ** 2) + eps)
    b = np.nan_to_num(b, nan=1.0, posinf=b_max, neginf=b_min)   # never propagate NaN/inf
    _BOUND_STATS["entries"] += b.size
    _BOUND_STATS["clipped_low"] += int(np.sum(b < b_min))
    _BOUND_STATS["clipped_high"] += int(np.sum(b > b_max))
    return np.clip(b, b_min, b_max)


def damped_diagonal_update(D, s, y, damping, lo, hi):
    """Damped elementwise secant estimate of the inverse Hessian diagonal
    (used by PBFGS): D_new = (1 - damping) * D + damping * clip(s / y).
    """
    ratio = np.zeros_like(D)
    safe = np.abs(y) > EPS
    ratio[safe] = s[safe] / y[safe]
    ratio = np.clip(ratio, lo, hi)
    return (1 - damping) * D + damping * ratio


def fletcher_reeves_beta(g_new, g_old, cap=1e6):
    denom = np.linalg.norm(g_old) ** 2
    if denom < EPS:
        return 0.0
    beta = (np.linalg.norm(g_new) ** 2) / denom
    if not np.isfinite(beta) or beta < 0 or beta > cap:
        return 0.0
    return beta


def armijo_backtrack(f_at, f_x, grad_dot_d, c1=1e-4, beta=0.5, max_iter=15, min_alpha=1e-12):
    """Backtracking line search. f_at(alpha) evaluates the objective at the
    trial step; returns the accepted step size alpha."""
    alpha = 1.0
    for _ in range(max_iter):
        if f_at(alpha) <= f_x + c1 * alpha * grad_dot_d:
            break
        alpha *= beta
        if alpha < min_alpha:
            break
    return alpha


def relative_step_change(W, W1, H, H1):
    """Stopping criterion, Eq. (56):

        sqrt(||W^{k+1}-W^k||_F^2 + ||H^{k+1}-H^k||_F^2)
        -------------------------------------------------  <= eps
        max{1, sqrt(||W^k||_F^2 + ||H^k||_F^2)}

    Shared by all four solvers (PSD, PCG, SPM, PBFGS) so every run is
    stopped on exactly the same, scale-invariant criterion.
    """
    step = np.sqrt(np.linalg.norm(W1 - W, "fro") ** 2 + np.linalg.norm(H1 - H, "fro") ** 2)
    current = np.sqrt(np.linalg.norm(W, "fro") ** 2 + np.linalg.norm(H, "fro") ** 2)
    return step / max(1.0, current)


class Diagnostics:
    """Collects per-iteration diagnostics and prints/saves them in a
    format shared by all four optimizers."""

    def __init__(self):
        self.grad_norm = []
        self.grad_norm_H = []
        self.f_factor = []
        self.f_orth = []
        self.rel_change = []

    def log(self, grad_norm, f_factor, f_orth, rel_change, grad_norm_H=None):
        self.grad_norm.append(grad_norm)
        self.grad_norm_H.append(np.nan if grad_norm_H is None else grad_norm_H)
        self.f_factor.append(f_factor)
        self.f_orth.append(f_orth)
        self.rel_change.append(rel_change)

    def save(self, prefix):
        np.save(f"{prefix}_grad_norm.npy", self.grad_norm)
        np.save(f"{prefix}_grad_norm_H.npy", self.grad_norm_H)
        np.save(f"{prefix}_f_factorisation.npy", self.f_factor)
        np.save(f"{prefix}_f_orthogonality.npy", self.f_orth)
        np.save(f"{prefix}_relative_change.npy", self.rel_change)

    def report(self, method_name, n_iter, W):
        print(f"--- {method_name}: optimization complete ---")
        print(f"Iterations         : {n_iter}")
        print(f"Gradient norm      : {self.grad_norm[-1]:.6e}")
        print(f"Factorisation loss : {self.f_factor[-1]:.6e}")
        print(f"Orthogonality loss : {self.f_orth[-1]:.6e}")
        print(f"Relative change    : {self.rel_change[-1]:.6e}")
        print(f"Orthogonality WtW  :\n{W.T @ W}\n")


# =====================================================================
# 5. Optimizers — standardized interface, SPM is the reference
# =====================================================================
def PSD_run(V, W, H, max_iter, tol=TOL, lam=None):
    lam = LAMBDA_SPARSE if lam is None else lam
    """Proximal Steepest Descent: direction = -gradient (no scaling).
    Baseline that the other three solvers are compared against."""
    diag = Diagnostics()
    gW, gH = df_W(V, W, H), df_H(V, W, H)

    it = 0
    for it in range(max_iter):
        dW = -gW
        alpha = armijo_backtrack(lambda a: f_W(V, W + a * dW, H), f_W(V, W, H), np.sum(gW * dW))
        W1 = np.maximum(W + alpha * dW, 0.0)

        dH = -gH
        alpha = armijo_backtrack(lambda a: f_H(V, W1, H + a * dH), f_H(V, W1, H), np.sum(gH * dH))
        # Variable-metric proximal map for lambda*||H||_1 with diagonal
        # inverse metric B_H: threshold = alpha*lambda*B_H elementwise.
        H1 = prox_l1_nonnegative(H + alpha * dH, alpha * lam)

        gW1, gH1 = df_W(V, W1, H1), df_H(V, W1, H1)
        rel_change = relative_step_change(W, W1, H, H1)
        f_fac = f_factorisation(V, W1, H1)
        diag.log(np.linalg.norm(gW1), f_fac, f_orthogonality(W1), rel_change, np.linalg.norm(gH1))

        W, H, gW, gH = W1, H1, gW1, gH1

        if rel_change < tol:  # Eq. (56)
            break
        if not np.isfinite(rel_change):
            print(f"[PSD] non-finite update at iteration {it}, stopping.")
            break

    diag.report("PSD", it + 1, W)
    return W, H, diag


def armijo_projected(f_of_point, point_at, x, fx, gx, c1=1e-4, beta=0.5, max_iter=30):
    """Projected Armijo backtracking: the sufficient-decrease test is done at
    the point that is actually used (after projection / prox), using the real
    displacement, so a step can never be accepted just because the *unprojected*
    trial looked good. Returns (alpha, new_point); alpha = 0 means no acceptable
    step was found and x is returned unchanged."""
    alpha = 1.0
    for _ in range(max_iter):
        xn = point_at(alpha)
        if np.isfinite(xn).all() and f_of_point(xn) <= fx + c1 * np.sum(gx * (xn - x)):
            return alpha, xn
        alpha *= beta
    return 0.0, x


def PCG_run(V, W, H, max_iter, tol=TOL, lam=None):
    """Proximal nonlinear Conjugate Gradient (Fletcher-Reeves).

    Safeguards: (i) line search on the projected point, (ii) restart to
    steepest descent if the CG direction is not a descent direction or no
    acceptable step exists, (iii) stop if even steepest descent fails.
    """
    lam = LAMBDA_SPARSE if lam is None else lam
    diag = Diagnostics()
    gW, gH = df_W(V, W, H), df_H(V, W, H)
    dW, dH = -gW, -gH

    it = 0
    for it in range(max_iter):
        # ---- W step -------------------------------------------------
        if np.sum(gW * dW) >= 0:
            dW = -gW
        fW = f_W(V, W, H)
        aW, W1 = armijo_projected(lambda Z: f_W(V, Z, H), lambda a: np.maximum(W + a * dW, 0.0), W, fW, gW)
        if aW == 0.0 and not np.array_equal(dW, -gW):
            dW = -gW
            aW, W1 = armijo_projected(lambda Z: f_W(V, Z, H), lambda a: np.maximum(W + a * dW, 0.0), W, fW, gW)

        # ---- H step (prox of lam*||H||_1 + nonnegativity) ------------
        if np.sum(gH * dH) >= 0:
            dH = -gH
        fH = f_H(V, W1, H)
        aH, H1 = armijo_projected(lambda Z: f_H(V, W1, Z), lambda a: prox_l1_nonnegative(H + a * dH, a * lam), H, fH, gH)
        if aH == 0.0 and not np.array_equal(dH, -gH):
            dH = -gH
            aH, H1 = armijo_projected(lambda Z: f_H(V, W1, Z), lambda a: prox_l1_nonnegative(H + a * dH, a * lam), H, fH, gH)

        if aW == 0.0 and aH == 0.0:
            print(f"[PCG] no acceptable step at iteration {it}, stopping.")
            break

        gW1, gH1 = df_W(V, W1, H1), df_H(V, W1, H1)
        dW = -gW1 + fletcher_reeves_beta(gW1, gW) * dW
        dH = -gH1 + fletcher_reeves_beta(gH1, gH) * dH

        rel_change = relative_step_change(W, W1, H, H1)
        f_fac = f_factorisation(V, W1, H1)
        diag.log(np.linalg.norm(gW1), f_fac, f_orthogonality(W1), rel_change, np.linalg.norm(gH1))

        W, H, gW, gH = W1, H1, gW1, gH1

        if not np.isfinite(rel_change):
            print(f"[PCG] non-finite update at iteration {it}, stopping.")
            break
        if rel_change < tol:  # Eq. (56)
            break

    diag.report("PCG", it + 1, W)
    return W, H, diag


def SPM_run(V, W, H, max_iter, tol=TOL, lam=None):
    lam = LAMBDA_SPARSE if lam is None else lam
    """Spectral Proximal Method (REFERENCE implementation).

    Direction: elementwise Barzilai-Borwein-type scaling of the negative
    gradient, re-estimated every iteration from the secant pair (s, y).
    """
    diag = Diagnostics()
    for k in _BOUND_STATS:
        _BOUND_STATS[k] = 0
    gW, gH = df_W(V, W, H), df_H(V, W, H)
    B_W, B_H = np.ones_like(W), np.ones_like(H)

    it = 0
    for it in range(max_iter):
        dW = -B_W * gW
        alpha = armijo_backtrack(lambda a: f_W(V, W + a * dW, H), f_W(V, W, H), np.sum(gW * dW))
        W1 = np.maximum(W + alpha * dW, 0.0)
        gW1 = df_W(V, W1, H)
        sW, yW = W1 - W, gW1 - gW
        if np.linalg.norm(sW) <= EPS or np.linalg.norm(yW) <= EPS:
            print(f"[SPM] degenerate secant pair for W at iteration {it}, stopping.")
            break
        B_W = spectral_metric(sW, yW)

        dH = -B_H * gH
        alpha = armijo_backtrack(lambda a: f_H(V, W1, H + a * dH), f_H(V, W1, H), np.sum(gH * dH))
        # Variable-metric proximal map for lambda*||H||_1 with diagonal
        # inverse metric B_H: threshold = alpha*lambda*B_H elementwise.
        H1 = prox_l1_nonnegative(H + alpha * dH, alpha * lam * B_H)
        gH1 = df_H(V, W1, H1)
        sH, yH = H1 - H, gH1 - gH
        if np.linalg.norm(sH) <= EPS or np.linalg.norm(yH) <= EPS:
            print(f"[SPM] degenerate secant pair for H at iteration {it}, stopping.")
            break
        B_H = spectral_metric(sH, yH)

        rel_change = relative_step_change(W, W1, H, H1)
        f_fac = f_factorisation(V, W1, H1)
        diag.log(np.linalg.norm(gW1), f_fac, f_orthogonality(W1), rel_change, np.linalg.norm(gH1))

        W, H, gW, gH = W1, H1, gW1, gH1

        if rel_change < tol:  # Eq. (56)
            break
        if not np.isfinite(rel_change):
            print(f"[SPM] non-finite update at iteration {it}, stopping.")
            break

    n = max(_BOUND_STATS["entries"], 1)
    print(f"[SPM] safeguard b in [{B_MIN:g}, {B_MAX:g}] active on "
          f"{100 * _BOUND_STATS['clipped_low'] / n:.4f}% (lower) / "
          f"{100 * _BOUND_STATS['clipped_high'] / n:.4f}% (upper) of entries")
    diag.report("SPM", it + 1, W)
    return W, H, diag


def PBFGS_run(V, W, H, max_iter, tol=TOL, lam=None, damping=0.2, scale_bounds=(1e-8, 1e8)):
    lam = LAMBDA_SPARSE if lam is None else lam
    """Proximal Diagonal-BFGS: elementwise diagonal inverse-Hessian
    estimate, updated every iteration via a damped secant rule (a cheap
    diagonal analogue of full BFGS)."""
    lo, hi = scale_bounds
    diag = Diagnostics()
    gW, gH = df_W(V, W, H), df_H(V, W, H)
    D_W, D_H = np.ones_like(W), np.ones_like(H)

    it = 0
    for it in range(max_iter):
        dW = -D_W * gW
        alpha = armijo_backtrack(
            lambda a: f_W(V, np.maximum(W + a * dW, 0.0), H), f_W(V, W, H), np.sum(gW * dW)
        )
        W1 = np.maximum(W + alpha * dW, 0.0)
        gW1 = df_W(V, W1, H)
        D_W = damped_diagonal_update(D_W, W1 - W, gW1 - gW, damping, lo, hi)

        dH = -D_H * gH
        alpha = armijo_backtrack(
            lambda a: f_H(V, W1, prox_l1_nonnegative(np.maximum(H + a * dH, 0.0), a * lam * D_H)),
            f_H(V, W1, H), np.sum(gH * dH),
        )
        H1 = prox_l1_nonnegative(np.maximum(H + alpha * dH, 0.0), alpha * lam * D_H)
        gH1 = df_H(V, W1, H1)
        D_H = damped_diagonal_update(D_H, H1 - H, gH1 - gH, damping, lo, hi)

        rel_change = relative_step_change(W, W1, H, H1)
        f_fac = f_factorisation(V, W1, H1)
        diag.log(np.linalg.norm(gW1), f_fac, f_orthogonality(W1), rel_change, np.linalg.norm(gH1))

        W, H, gW, gH = W1, H1, gW1, gH1

        if rel_change < tol:  # Eq. (56)
            break
        if not np.isfinite(rel_change):
            print(f"[PBFGS] non-finite update at iteration {it}, stopping.")
            break

    diag.report("PBFGS", it + 1, W)
    return W, H, diag


OPTIMIZERS = {0: PSD_run, 1: PCG_run, 2: SPM_run, 3: PBFGS_run}


# ---- lightweight, always-on evaluation counters ---------------------
# (replaces cProfile: no overhead, works in IDEs/Python 3.12+, and counts
# something meaningful: objective and gradient evaluations, incl. line-search trials)
_EVALS = {"objective": 0, "gradient": 0}


def _counted(kind, fn):
    def wrapper(*args, **kwargs):
        _EVALS[kind] += 1
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


f_W = _counted("objective", f_W)
f_H = _counted("objective", f_H)
df_W = _counted("gradient", df_W)
df_H = _counted("gradient", df_H)


def profile_run(func, *args, **kwargs):
    """Run `func`; return (result, {"objective": n_f, "gradient": n_g}).
    Kept under the old name so existing call sites keep working."""
    before = dict(_EVALS)
    result = func(*args, **kwargs)
    return result, {k: _EVALS[k] - before[k] for k in _EVALS}


def call_fields(evals=None, extra_gradient=0):
    """Columns describing computational cost.
    evals          : dict from profile_run (proximal solvers)
    extra_gradient : gradient-equivalent evaluations done outside the solver
                     (each multiplicative-update iteration = 2: one W, one H).
    num_function_calls = objective evals + gradient evals."""
    evals = evals or {"objective": 0, "gradient": 0}
    n_obj, n_grad = evals["objective"], evals["gradient"] + extra_gradient
    return {"num_objective_evals": n_obj, "num_gradient_evals": n_grad,
            "num_function_calls": n_obj + n_grad}


# =====================================================================
# 6. Topic quality metrics
# =====================================================================
# NOTE ON NAMING: the two functions below measure how DISTINCT topics are
# from one another (diversity / redundancy across topics), NOT whether a
# single topic's top words make semantic sense together. That second
# property is "coherence" in the topic-modeling literature and is measured
# separately below via corpus co-occurrence statistics (UMass / NPMI).
# Do not report these as "coherence" in tables/figures.

def topic_redundancy_jaccard(top_terms_per_topic):
    """Average pairwise Jaccard overlap between topics' top-term sets.
    Higher = more redundant/overlapping topics (worse). Was previously
    named `calculate_topic_coherence`; renamed because it measures
    inter-topic redundancy, not within-topic coherence."""
    score, n_pairs = 0.0, 0
    for i in range(len(top_terms_per_topic)):
        for j in range(i + 1, len(top_terms_per_topic)):
            a, b = set(top_terms_per_topic[i]), set(top_terms_per_topic[j])
            if a | b:
                score += len(a & b) / len(a | b)
                n_pairs += 1
    return score / n_pairs if n_pairs else 0.0


def topic_redundancy_cosine(H_normalized):
    """Average pairwise cosine similarity between topics' term-weight
    vectors (rows of H), averaged across topic pairs. Higher = more
    redundant topics (worse). Was previously named
    `calculate_cosine_similarity` and mislabeled as "coherence"; it
    measures topic diversity/redundancy, not within-topic coherence."""
    sim = cosine_similarity(H_normalized)
    np.fill_diagonal(sim, 0)
    n = sim.shape[0]
    return float(sim.sum() / (n * (n - 1))) if n > 1 else 0.0


# Backward-compatible aliases (old names), so existing analysis notebooks
# that import the old function names don't break; new code should use the
# renamed versions above.
calculate_topic_coherence = topic_redundancy_jaccard
calculate_cosine_similarity = topic_redundancy_cosine


def build_cooccurrence(binary_doc_term):
    """From a (docs x vocab) binary presence matrix, return per-word
    document frequency and the word-word co-document-frequency matrix
    needed for UMass/NPMI coherence."""
    doc_freq = np.asarray(binary_doc_term.sum(axis=0)).flatten()
    cooc = np.asarray((binary_doc_term.T @ binary_doc_term).todense()
                       if hasattr(binary_doc_term, "todense") else binary_doc_term.T @ binary_doc_term)
    return doc_freq, cooc


def _top_word_indices(H, n_top_words):
    return [topic.argsort()[::-1][:n_top_words] for topic in np.asarray(H)]


def coherence_umass(H, doc_freq, cooc, n_top_words=10, eps=1.0):
    """UMass coherence (Mimno et al., 2011), averaged over topics: for each
    topic's top words, sum log((D(w_i, w_j) + eps) / D(w_j)) over ordered
    pairs i>j. Less negative (closer to 0) is better."""
    topic_scores = []
    for idx in _top_word_indices(H, n_top_words):
        s, n = 0.0, 0
        for i in range(1, len(idx)):
            for j in range(i):
                s += np.log((cooc[idx[i], idx[j]] + eps) / (doc_freq[idx[j]] + eps))
                n += 1
        topic_scores.append(s / n if n else 0.0)
    return float(np.mean(topic_scores))


def coherence_npmi(H, doc_freq, cooc, n_docs, n_top_words=10, eps=1e-12):
    """Normalized PMI coherence (Bouma, 2009; Röder et al., 2015),
    averaged over topics and word pairs. Range [-1, 1]; higher is better."""
    topic_scores = []
    for idx in _top_word_indices(H, n_top_words):
        s, n = 0.0, 0
        for i in range(1, len(idx)):
            for j in range(i):
                p_ij = cooc[idx[i], idx[j]] / n_docs
                p_i = doc_freq[idx[i]] / n_docs
                p_j = doc_freq[idx[j]] / n_docs
                if p_ij > 0:
                    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
                    npmi = pmi / (-np.log(p_ij + eps))
                else:
                    npmi = -1.0  # convention for zero co-occurrence
                s += npmi
                n += 1
        topic_scores.append(s / n if n else 0.0)
    return float(np.mean(topic_scores))


# =====================================================================
# 7. Plotting
# =====================================================================
import re as _re
def _safe_name(path):
    p = Path(path)
    return p.with_name(_re.sub(r'[^A-Za-z0-9._-]', '_', p.name))


def safe_savefig(fig, save_path, **kw):
    """Save a figure without letting a file-system error kill the run
    (Windows can raise Errno 22 if a PNG is locked by a viewer/OneDrive)."""
    import time as _t
    target = _safe_name(save_path)
    alt = target.with_name(f"{target.stem}_{int(_t.time())}{target.suffix}")
    for attempt, tgt in enumerate([target, alt]):
        try:
            fig.savefig(str(tgt), **kw)
            return str(tgt)
        except OSError as e:
            print(f"[warn] could not save {tgt} ({e}); "
                  f"{'retrying with new name' if attempt == 0 else 'skipping'}")
    return None


def plot_top_words(H, feature_names, n_top_words, title, save_path):
    n_components = H.shape[0]
    n_cols = int(np.ceil(n_components / 5))
    fig, axes = plt.subplots(5, n_cols, figsize=(20, 30), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for topic_idx, topic in enumerate(np.asarray(H)):
        top_idx = topic.argsort()[::-1][:n_top_words]
        top_features = [feature_names[i] for i in top_idx]
        weights = topic[top_idx]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=24)
        ax.invert_yaxis()
        ax.tick_params(labelsize=16)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    for i in range(n_components, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=36)
    plt.subplots_adjust(top=0.92, bottom=0.05, wspace=0.7, hspace=0.3)
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wordclouds(H, feature_names, n_top_words, title, save_path):
    n_components = H.shape[0]
    n_cols = int(np.ceil(n_components / 5))
    fig, axes = plt.subplots(5, n_cols, figsize=(20, 30))
    axes = np.atleast_1d(axes).flatten()

    for topic_idx, topic_weights in enumerate(H):
        if (not np.isfinite(topic_weights).all()) or topic_weights.max() <= 0:
            ax = axes[topic_idx]
            ax.text(0.5, 0.5, "empty topic", ha="center", va="center", fontsize=20)
            ax.axis("off")
            ax.set_title(f"Topic {topic_idx + 1}", fontsize=20)
            continue
        top_idx = topic_weights.argsort()[-n_top_words:][::-1]
        word_freq = {feature_names[i]: float(topic_weights[i]) for i in top_idx}

        wc = WordCloud(
            background_color="white", width=800, height=400,
            max_words=n_top_words, colormap="viridis",
        ).generate_from_frequencies(word_freq)

        ax = axes[topic_idx]
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=20)

    for i in range(n_components, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=36)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curve(values, ylabel, title, save_path):
    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_orthogonality_heatmap(W, save_path):
    fig = plt.figure()
    heatmap = sn.heatmap(data=W.T @ W, cmap="mako")
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_pattern(H_normalized, save_path):
    fig = plt.figure()
    binary = (H_normalized > 0).astype(float)
    plt.pcolor(binary, cmap="Greys")
    plt.xlabel("Vocabulary term")
    plt.ylabel("Topic")
    plt.tight_layout()
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_optimizer_comparison(histories, metric, ylabel, title, save_path,
                              log_scale=False, methods=None):
    """Compare all requested methods on a common metric/history axis."""
    methods = methods or [m for m in ALL_METHODS if m in histories]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        values = histories.get(method, [])
        if values is not None and len(values):
            ax.plot(np.arange(1, len(values) + 1), values, label=method,
                    linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    plt.tight_layout()
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_optimizer_metric_bars(summary, metric, ylabel, title, save_path,
                               lower_is_better=False, methods=None):
    """Bar chart of mean +/- SD across repeated seeds for all methods."""
    methods = methods or [m for m in ALL_METHODS if m in summary["method"].values]
    means, stds, labels = [], [], []
    for method in methods:
        rows = summary.loc[summary["method"] == method]
        if rows.empty or f"{metric}_mean" not in rows.columns:
            continue
        row = rows.iloc[0]
        means.append(row[f"{metric}_mean"])
        stds.append(row[f"{metric}_std"])
        labels.append(method)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    safe_savefig(fig, save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_words_comparison(method_outputs, feature_names, n_top_words,
                              n_components, tag, out_dir):
    """Save one top-word figure for every method."""
    for method, output in method_outputs.items():
        H = normalize_rows(output["H"])
        plot_top_words(
            H, feature_names, n_top_words,
            f"Top words - {method} ({n_components} topics)",
            str(out_dir / f"top_words_{method}_{tag}.png"),
        )


def plot_wordcloud_comparison(method_outputs, feature_names, n_top_words,
                              n_components, tag, out_dir):
    """Save one word-cloud figure for every method."""
    for method, output in method_outputs.items():
        H = normalize_rows(output["H"])
        plot_wordclouds(
            H, feature_names, n_top_words,
            f"Word clouds - {method} ({n_components} topics)",
            str(out_dir / f"wordclouds_{method}_{tag}.png"),
        )


# 8. Evaluation, repeated-seed experiments, and sensitivity study
# =====================================================================
def normalize_rows(H):
    H = np.asarray(H, dtype=float).copy()
    H[H < 0] = 0
    row_sum = H.sum(axis=1, keepdims=True)
    return H / np.maximum(row_sum, EPS)


def hoyer_sparsity(H):
    """Scale-invariant Hoyer sparsity, averaged over topic rows.

    0 = dense/equal-magnitude row; 1 = maximally sparse row.
    Unlike exact-zero counting, this remains informative for multiplicative
    NMF methods, whose positive iterates normally contain no exact zeros.
    """
    H = np.maximum(np.asarray(H, dtype=float), 0.0)
    n = H.shape[1]
    if n <= 1:
        return 0.0
    l1 = np.sum(H, axis=1)
    l2 = np.linalg.norm(H, axis=2 if H.ndim > 2 else 1)
    values = np.ones(H.shape[0])
    safe = l2 > EPS
    values[safe] = (np.sqrt(n) - l1[safe] / l2[safe]) / (np.sqrt(n) - 1.0)
    return float(np.mean(np.clip(values, 0.0, 1.0)))


def column_normalized_orthogonality(W):
    """Scale-invariant orthogonality error of W.

    The raw ||W^T W-I||_F is affected by NMF's scale ambiguity. This metric
    first normalizes each W column to unit norm and then measures the
    off-diagonal/orthonormality error.
    """
    W = np.asarray(W, dtype=float)
    norms = np.linalg.norm(W, axis=0)
    safe = norms > EPS
    Wn = W.copy()
    Wn[:, safe] /= norms[safe]
    G = Wn.T @ Wn
    return float(np.linalg.norm(G - np.eye(G.shape[0]), "fro") ** 2)


def binarize_doc_term(tfidf):
    """Binary (doc x vocab) presence matrix, used only for coherence
    co-occurrence statistics (not for fitting)."""
    binary = tfidf.copy()
    binary.data = np.ones_like(binary.data)
    return binary


def storage_metrics(H, value_bytes=8, index_bytes=4):
    """Storage cost of H if kept in CSR format (float64 values + int32 column
    indices + int32 row pointers) versus dense float64. Only EXACT zeros can be
    dropped, so tiny-but-nonzero entries (e.g. from multiplicative updates)
    do not save anything. CSR beats dense only if nnz < ~2/3 of entries,
    i.e. more than ~1/3 exact zeros."""
    H = np.asarray(H)
    nnz = int(np.count_nonzero(H))
    dense = H.size * value_bytes
    csr = nnz * (value_bytes + index_bytes) + (H.shape[0] + 1) * index_bytes
    return {
        "exact_zero_fraction": float(1.0 - nnz / H.size),
        "nnz_H": nnz,
        "storage_H_dense_bytes": dense,
        "storage_H_csr_bytes": csr,
        "storage_H_ratio": csr / dense,        # < 1 means sparse storage saves memory
    }


def evaluate_topics(H, W, doc_freq, cooc, n_docs, n_top_words=10):
    """Common evaluation metrics for all eight methods.

    `zero_fraction` reports literal zeros, while `hoyer_sparsity` is the
    preferred scale-invariant sparsity measure for multiplicative NMF.
    `orthogonality_normalized` removes NMF's W/H scale ambiguity; the raw
    objective penalty is reported separately as `function_value_orthogonality`.
    """
    H_norm = normalize_rows(H)
    return {
        **storage_metrics(H),
        "zero_fraction": float(np.mean(H <= EPS)),
        "hoyer_sparsity": hoyer_sparsity(H),
        "orthogonality_normalized": column_normalized_orthogonality(W),
        "topic_redundancy_cosine": topic_redundancy_cosine(H_norm),
        "topic_redundancy_jaccard": topic_redundancy_jaccard(_top_word_indices(H_norm, n_top_words)),
        "coherence_umass": coherence_umass(H_norm, doc_freq, cooc, n_top_words),
        "coherence_npmi": coherence_npmi(H_norm, doc_freq, cooc, n_docs, n_top_words),
    }


def run_single_baseline(name, tfidf, n_components, max_iter, init_mode,
                       doc_freq, cooc, n_docs, seed=None):
    """Run one baseline using the same initialization seed as the proximal methods."""
    if seed is not None:
        np.random.seed(seed)
    fn = BASELINE_METHODS[name]
    t0 = time()
    W, H, history = fn(tfidf, n_components, max_iter, init_mode)
    elapsed = time() - t0
    V_dense = to_dense_array(tfidf)
    metrics = evaluate_topics(H, W, doc_freq, cooc, n_docs)
    metrics.update({
        "method": name,
        "final_loss": history[-1] if history else np.nan,
        "factorisation_loss": f_factorisation(V_dense, W, H),
        "function_value_orthogonality": f_orthogonality(W),
        "gradient_norm": np.nan,
        "gradient_norm_H": np.nan,
        "relative_change": np.nan,
        "iterations": len(history),
        "elapsed_seconds": elapsed,
        **call_fields(None, extra_gradient=2 * len(history)),
    })
    return W, H, metrics, history


def run_single_optimizer(opt_method, tfidf, n_components, max_itr, MAXITER, init_mode, update_rule,
                          doc_freq, cooc, n_docs):
    warm_start_fn = ed_method if update_rule == 0 else dvg_method
    if max_itr == 0:
        W, H = initialize(tfidf, n_components, init_mode)
    else:
        W, H, _ = warm_start_fn(tfidf, n_components, max_itr, init_mode)

    # PSD/PCG/SPM/PBFGS all compute V - W@H every iteration; W@H is dense,
    # so if V stays sparse that subtraction silently returns a
    # numpy.matrix (see `to_dense_array` docstring) and later elementwise
    # ops crash as matrix products. Densify once, here, before handing V
    # to any of the four proximal-gradient solvers.
    V_dense = to_dense_array(tfidf)

    method_name = METHOD_NAMES[opt_method]
    optimizer_fn = OPTIMIZERS[opt_method]
    t0 = time()
    (W_opt, H_opt, diag), ncalls = profile_run(optimizer_fn, V_dense, W, H, MAXITER)
    elapsed = time() - t0

    metrics = evaluate_topics(H_opt, W_opt, doc_freq, cooc, n_docs)
    metrics.update({
        "method": method_name,
        "final_loss": diag.f_factor[-1],
        "factorisation_loss": diag.f_factor[-1],
        "function_value_orthogonality": diag.f_orth[-1],
        "gradient_norm": diag.grad_norm[-1],
        "relative_change": diag.rel_change[-1],
        "iterations": len(diag.grad_norm),
        "elapsed_seconds": elapsed,
        **call_fields(ncalls, extra_gradient=2 * max_itr),
    })
    return W_opt, H_opt, metrics, diag


def _metrics_from_optimizer(method_name, W, H, diag, elapsed, ncalls,
                             doc_freq, cooc, n_docs, warm_iters=0):
    """Build one comparable metrics row for a proximal-gradient solver."""
    metrics = evaluate_topics(H, W, doc_freq, cooc, n_docs)
    metrics.update({
        "method": method_name,
        "final_loss": diag.f_factor[-1] if diag.f_factor else np.nan,
        "factorisation_loss": diag.f_factor[-1] if diag.f_factor else np.nan,
        "function_value_orthogonality": diag.f_orth[-1] if diag.f_orth else np.nan,
        "gradient_norm": diag.grad_norm[-1] if diag.grad_norm else np.nan,
        "gradient_norm_H": diag.grad_norm_H[-1] if diag.grad_norm_H else np.nan,
        "relative_change": diag.rel_change[-1] if diag.rel_change else np.nan,
        "iterations": len(diag.grad_norm),
        "elapsed_seconds": elapsed,
        **call_fields(ncalls, extra_gradient=2 * warm_iters),
    })
    return metrics


def run_all_optimizers(tfidf, n_components, max_itr, MAXITER, init_mode,
                       update_rule, doc_freq, cooc, n_docs):
    """Run PSD, PCG, SPM and PBFGS from the SAME warm-start W,H.

    This is important for a fair comparison: within each seed, all four
    methods receive identical data, initialization and warm-start iterations.
    """
    warm_start_fn = ed_method if update_rule == 0 else dvg_method

    if max_itr == 0:
        W0, H0 = initialize(tfidf, n_components, init_mode)
    else:
        W0, H0, _ = warm_start_fn(
            tfidf, n_components, max_itr, init_mode
        )

    V_dense = to_dense_array(tfidf)
    outputs = {}

    # Copy the same starting point for every optimizer.
    for opt_id in sorted(OPTIMIZERS):
        method_name = METHOD_NAMES[opt_id]
        optimizer_fn = OPTIMIZERS[opt_id]

        W_init = W0.copy()
        H_init = H0.copy()

        t0 = time()
        (W_opt, H_opt, diag), ncalls = profile_run(
            optimizer_fn, V_dense, W_init, H_init, MAXITER
        )
        elapsed = time() - t0

        metrics = _metrics_from_optimizer(
            method_name, W_opt, H_opt, diag, elapsed, ncalls,
            doc_freq, cooc, n_docs,
            warm_iters=max_itr
        )
        outputs[method_name] = {
            "W": W_opt,
            "H": H_opt,
            "diag": diag,
            "metrics": metrics,
        }

    return outputs


def run_experiment(n_components, update_rule, init_mode, max_itr, MAXITER,
                   opt_method=None, seeds=(0, 1, 2, 3, 4),
                   include_baselines=True, make_plots=True,
                   output_dir="proximal_comparison"):
    """Benchmark all eight methods from a common starting point.

    Methods:
        PSD, SPM, PBFGS, PCG, NMF-ED, NMF-KL, ONMF, SparseNMF.

    For each seed, one initialization (and optional warm-start) is created
    once and copied to every method. This removes the previous confounding
    effect where the proximal methods started from an NMF solution while
    baselines independently started from random/NNDSVD factors.
    """
    update_rule_name = UPDATE_RULE_NAMES[update_rule]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess(N_SAMPLES)
    tfidf, vectorizer = extract_tfidf(df["text"], N_FEATURES)
    feature_names = vectorizer.get_feature_names_out()
    V_dense = to_dense_array(tfidf)

    binary = binarize_doc_term(tfidf)
    doc_freq, cooc = build_cooccurrence(binary)
    n_docs = tfidf.shape[0]
    tag = f"{n_components}topics_{update_rule_name}_init{init_mode}"

    rows = []
    representative_outputs = None
    representative_histories = None

    for seed in seeds:
        np.random.seed(seed)

        # ONE raw initialisation shared by every method.
        W_init, H_init = initialize(V_dense, n_components, init_mode)

        # Proximal solvers only: warm-start with max_itr NMF iterations
        # beginning at that SAME raw initialisation.
        warm_hist = []
        if max_itr > 0:
            warm_start_fn = ed_method if update_rule == 0 else dvg_method
            W0, H0, warm_hist = warm_start_fn(
                V_dense, n_components, max_itr, init_mode, init=(W_init, H_init)
            )
        else:
            W0, H0 = W_init.copy(), H_init.copy()

        baseline_iters = (max_itr + MAXITER) if EQUAL_ITERATION_BUDGET else MAXITER

        all_outputs = {}

        if include_baselines:
            for name in BASELINE_METHODS:
                t0 = time()
                W_b, H_b, history_b = _baseline_from_start(
                    name, V_dense, W_init, H_init, baseline_iters, l1_h=BASELINE_L1
                )
                elapsed = time() - t0
                metrics_b = evaluate_topics(H_b, W_b, doc_freq, cooc, n_docs)
                metrics_b.update({
                    "method": name,
                    "final_loss": history_b[-1] if history_b else np.nan,
                    "factorisation_loss": f_factorisation(V_dense, W_b, H_b),
                    "function_value_orthogonality": f_orthogonality(W_b),
                    "gradient_norm": np.nan,
                    "gradient_norm_H": np.nan,
                    "relative_change": np.nan,
                    "iterations": len(history_b),
                    "elapsed_seconds": elapsed,
                    **call_fields(None, extra_gradient=2 * len(history_b)),
                    "seed": seed,
                })
                rows.append(metrics_b)
                all_outputs[name] = {
                    "W": W_b, "H": H_b, "history": history_b,
                    "metrics": metrics_b, "diag": None
                }

        # All four proximal methods get exactly the same W0,H0.
        for opt_id in sorted(OPTIMIZERS):
            method_name = METHOD_NAMES[opt_id]
            optimizer_fn = OPTIMIZERS[opt_id]
            t0 = time()
            (W_opt, H_opt, diag), ncalls = profile_run(
                optimizer_fn, V_dense, W0.copy(), H0.copy(), MAXITER
            )
            elapsed = time() - t0

            metrics = _metrics_from_optimizer(
                method_name, W_opt, H_opt, diag, elapsed, ncalls,
                doc_freq, cooc, n_docs,
                warm_iters=max_itr
            )
            metrics["seed"] = seed
            rows.append(metrics)
            all_outputs[method_name] = {
                "W": W_opt, "H": H_opt,
                "history": list(warm_hist) + list(diag.f_factor),
                "metrics": metrics, "diag": diag
            }

        # Use seed 0 as the representative qualitative run. Do not select a
        # seed by the proposed method's NPMI, which can bias visual reporting.
        if representative_outputs is None:
            representative_outputs = all_outputs
            representative_histories = {
                m: all_outputs[m]["history"]
                for m in ALL_METHODS if m in all_outputs
            }

    results = pd.DataFrame(rows)
    method_order = {m: i for i, m in enumerate(ALL_METHODS)}
    results["method_order"] = results["method"].map(method_order)
    results = results.sort_values(["seed", "method_order"]).drop(columns="method_order")
    results = results.reset_index(drop=True)

    numeric_cols = results.select_dtypes(include=[np.number]).columns.drop("seed", errors="ignore")
    agg = results.groupby("method")[numeric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    agg["method_order"] = agg["method"].map(method_order)
    agg = agg.sort_values("method_order").drop(columns="method_order")

    raw_path = out_dir / f"experiment_raw_all_methods_{tag}.csv"
    summary_path = out_dir / f"all_methods_comparison_{tag}.csv"
    results.to_csv(raw_path, index=False)
    agg.to_csv(summary_path, index=False)

    table_columns = [
        "method",
        "factorisation_loss_mean", "factorisation_loss_std",
        "function_value_orthogonality_mean", "function_value_orthogonality_std",
        "orthogonality_normalized_mean", "orthogonality_normalized_std",
        "gradient_norm_mean", "gradient_norm_std",
        "gradient_norm_H_mean", "gradient_norm_H_std",
        "relative_change_mean", "relative_change_std",
        "iterations_mean", "iterations_std",
        "elapsed_seconds_mean", "elapsed_seconds_std",
        "hoyer_sparsity_mean", "hoyer_sparsity_std",
        "zero_fraction_mean", "zero_fraction_std",
        "exact_zero_fraction_mean", "exact_zero_fraction_std",
        "storage_H_ratio_mean", "storage_H_ratio_std",
        "coherence_umass_mean", "coherence_umass_std",
        "coherence_npmi_mean", "coherence_npmi_std",
        "topic_redundancy_cosine_mean", "topic_redundancy_cosine_std",
        "topic_redundancy_jaccard_mean", "topic_redundancy_jaccard_std",
    ]
    table_columns = [c for c in table_columns if c in agg.columns]
    publication_table = agg[table_columns]
    publication_path = out_dir / f"all_methods_publication_table_{tag}.csv"
    publication_table.to_csv(publication_path, index=False)

    print("\n=== ALL METHODS: Mean +/- SD over", len(seeds), "seeds ===")
    print(agg.to_string(index=False))
    print(f"\nRaw results: {raw_path}")
    print(f"Comparison table: {summary_path}")
    print(f"Publication table: {publication_path}")

    if make_plots and representative_outputs is not None:
        plot_optimizer_comparison(
            representative_histories, "history",
            r"Factorisation loss $\|V-WH\|_F^2$",
            f"All-method convergence comparison - {n_components} topics",
            str(out_dir / f"comparison_all_methods_factorisation_{tag}.png"),
            log_scale=True, methods=ALL_METHODS
        )

        for metric, ylabel, title in [
            ("factorisation_loss", "Final factorisation loss",
             "Final factorisation loss: all methods (mean +/- SD)"),
            ("iterations", "Iterations",
             "Iterations: all methods (mean +/- SD)"),
            ("elapsed_seconds", "Time (seconds)",
             "Runtime: all methods (mean +/- SD)"),
            ("hoyer_sparsity", "Hoyer sparsity of H",
             "H sparsity (Hoyer): all methods (mean +/- SD)"),
            ("zero_fraction", "Exact-zero fraction of H",
             "Exact zeros in H: all methods (mean +/- SD)"),
            ("coherence_npmi", "NPMI",
             "NPMI coherence: all methods (mean +/- SD)"),
            ("coherence_umass", "UMass",
             "UMass coherence: all methods (mean +/- SD)"),
            ("orthogonality_normalized", r"$\|\widetilde W^T\widetilde W-I\|_F^2$",
             "Scale-normalized W orthogonality: all methods (mean +/- SD)"),
            ("function_value_orthogonality", r"$\|W^TW-I\|_F^2$",
             "Raw W orthogonality penalty: all methods (mean +/- SD)"),
        ]:
            if f"{metric}_mean" in agg.columns:
                plot_optimizer_metric_bars(
                    agg, metric, ylabel, title,
                    str(out_dir / f"bar_all_methods_{metric}_{tag}.png"),
                    methods=ALL_METHODS
                )

        proximal_names = list(METHOD_NAMES.values())
        proximal_outputs = {
            m: representative_outputs[m] for m in proximal_names
        }
        plot_optimizer_comparison(
            {m: o["diag"].grad_norm for m, o in proximal_outputs.items()},
            "history", r"$\|\nabla_W f(W,H)\|_F$",
            f"Proximal methods: W-gradient norm - {n_components} topics",
            str(out_dir / f"comparison_proximal_gradient_W_{tag}.png"),
            log_scale=True, methods=proximal_names
        )
        plot_optimizer_comparison(
            {m: o["diag"].grad_norm_H for m, o in proximal_outputs.items()},
            "history", r"$\|\nabla_H f(W,H)\|_F$",
            f"Proximal methods: H-gradient norm - {n_components} topics",
            str(out_dir / f"comparison_proximal_gradient_H_{tag}.png"),
            log_scale=True, methods=proximal_names
        )
        plot_optimizer_comparison(
            {m: o["diag"].rel_change for m, o in proximal_outputs.items()},
            "history", "Relative step change",
            f"Proximal methods: stopping criterion - {n_components} topics",
            str(out_dir / f"comparison_proximal_relative_change_{tag}.png"),
            log_scale=True, methods=proximal_names
        )

        plot_top_words_comparison(
            representative_outputs, feature_names, N_TOP_WORDS,
            n_components, tag, out_dir
        )
        plot_wordcloud_comparison(
            representative_outputs, feature_names, N_TOP_WORDS,
            n_components, tag, out_dir
        )
        for method, output in representative_outputs.items():
            H_norm = normalize_rows(output["H"])
            plot_orthogonality_heatmap(
                output["W"], str(out_dir / f"orthogonality_{method}_{tag}.png")
            )
            plot_sparsity_pattern(
                H_norm, str(out_dir / f"sparsity_{method}_{tag}.png")
            )

    return results, agg


def sensitivity_study(n_components=10, update_rule=0, init_mode=0, max_itr=20, MAXITER=50,
                       opt_method=2, mu_grid=(1.0, 10.0, 50.0), lam_grid=(1e-4, 1e-3, 1e-2)):
    """Sweeps the orthogonality weight MU and sparsity weight LAMBDA_SPARSE
    on a small grid, reporting final loss / coherence / sparsity for each
    combination. Uses the module-level MU/LAMBDA_SPARSE as the mutated
    globals since f_W/df_W etc. read them directly; restores originals
    afterward.
    """
    global MU, LAMBDA_SPARSE
    orig_mu, orig_lambda = MU, LAMBDA_SPARSE

    df = load_and_preprocess(N_SAMPLES)
    tfidf, _ = extract_tfidf(df["text"], N_FEATURES)
    binary = binarize_doc_term(tfidf)
    doc_freq, cooc = build_cooccurrence(binary)
    n_docs = tfidf.shape[0]

    rows = []
    try:
        for mu in mu_grid:
            for lam in lam_grid:
                MU, LAMBDA_SPARSE = mu, lam
                np.random.seed(0)
                _, _, metrics, _ = run_single_optimizer(
                    opt_method, tfidf, n_components, max_itr, MAXITER, init_mode, update_rule,
                    doc_freq, cooc, n_docs)
                metrics.update({"mu": mu, "lambda_sparse": lam})
                rows.append(metrics)
    finally:
        MU, LAMBDA_SPARSE = orig_mu, orig_lambda

    table = pd.DataFrame(rows)
    table.to_csv("sensitivity_study.csv", index=False)
    print(table[["mu", "lambda_sparse", "final_loss", "coherence_npmi", "sparsity"]].to_string(index=False))
    return table


CODE_DATA_AVAILABILITY = """
Code and data availability
---------------------------
Data: all experiments use the public 20-newsgroups corpus, retrieved via
`sklearn.datasets.fetch_20newsgroups` (scikit-learn >= 1.0). No
redistribution of raw data is required; the corpus is downloaded on
first run and cached locally by scikit-learn.

Code: the full implementation (preprocessing, baselines, and all four
proximal-gradient solvers) is contained in this single script,
`nmf_proximal_methods_all_comparison.py`. For submission, archive a tagged release of
the containing repository on Zenodo or OSF to obtain a citable DOI,
rather than an "available on request" statement. Report the exact
package versions used (numpy, scipy, scikit-learn, nltk, wordcloud) via
a `requirements.txt` / `environment.yml` alongside the script, and the
random seeds used for the reported runs (see `run_experiment(seeds=...)`).
"""


# =====================================================================
# 9. CLI entry point
# =====================================================================
def main():
    """CLI usage:
        python nmf_proximal_methods.py <max_itr> <MAXITER> <init_mode> \
            <update_rule> <opt_method> <n_components> [n_seeds]

    n_seeds (optional, default 5): number of random seeds to repeat the
    full experiment over ALL EIGHT methods (PSD, SPM, PBFGS, PCG, NMF-ED, NMF-KL, ONMF, SparseNMF),
    reported as mean +/- sd. The opt_method argument is retained for
    backward compatibility but no longer selects a single method.
    """
    max_itr = int(sys.argv[1])
    MAXITER = int(sys.argv[2])
    init_mode = int(sys.argv[3])
    update_rule = int(sys.argv[4])
    opt_method = int(sys.argv[5])
    n_components = int(sys.argv[6])
    n_seeds = int(sys.argv[7]) if len(sys.argv) > 7 else 5

    print(CODE_DATA_AVAILABILITY)
    run_experiment(
        n_components=n_components, update_rule=update_rule, init_mode=init_mode,
        max_itr=max_itr, MAXITER=MAXITER, opt_method=opt_method,
        seeds=tuple(range(n_seeds)), include_baselines=True, make_plots=True,
    )


if __name__ == "__main__":
    main()