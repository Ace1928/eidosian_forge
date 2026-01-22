import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
def _initialize_nmf(X, n_components, init=None, eps=1e-06, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, 'NMF initialization')
    n_samples, n_features = X.shape
    if init is not None and init != 'random' and (n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when n_components <= min(n_samples, n_features)".format(init))
    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvda'
        else:
            init = 'random'
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(X.dtype, copy=False)
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(X.dtype, copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return (W, H)
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
    for j in range(1, n_components):
        x, y = (U[:, j], V[j, :])
        x_p, y_p = (np.maximum(x, 0), np.maximum(y, 0))
        x_n, y_n = (np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0)))
        x_p_nrm, y_p_nrm = (norm(x_p), norm(y_p))
        x_n_nrm, y_n_nrm = (norm(x_n), norm(y_n))
        m_p, m_n = (x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm)
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    W[W < eps] = 0
    H[H < eps] = 0
    if init == 'nndsvd':
        pass
    elif init == 'nndsvda':
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == 'nndsvdar':
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError('Invalid init parameter: got %r instead of one of %r' % (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))
    return (W, H)