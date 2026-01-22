import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
def _expm(A, use_exact_onenorm):
    if isinstance(A, (list, tuple, np.matrix)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    if A.shape == (0, 0):
        out = np.zeros([0, 0], dtype=A.dtype)
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)
        return out
    if A.shape == (1, 1):
        out = [[np.exp(A[0, 0])]]
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)
        return np.array(out)
    if (isinstance(A, np.ndarray) or issparse(A) or is_pydata_spmatrix(A)) and (not np.issubdtype(A.dtype, np.inexact)):
        A = A.astype(float)
    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None
    if use_exact_onenorm == 'auto':
        use_exact_onenorm = A.shape[0] < 200
    h = _ExpmPadeHelper(A, structure=structure, use_exact_onenorm=use_exact_onenorm)
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 0.01495585217958292 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V, structure=structure)
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 0.253939833006323 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V, structure=structure)
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 0.9504178996162932 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V, structure=structure)
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    if eta_5 == 0:
        s = 0
    else:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2 ** (-s) * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        X = _fragment_2_1(X, h.A, s)
    else:
        for i in range(s):
            X = X.dot(X)
    return X