import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _logm(A):
    """
    Compute the matrix logarithm.

    See the logm docstring in matfuncs.py for more info.

    Notes
    -----
    In this function we look at triangular matrices that are similar
    to the input matrix. If any diagonal entry of such a triangular matrix
    is exactly zero then the original matrix is singular.
    The matrix logarithm does not exist for such matrices,
    but in such cases we will pretend that the diagonal entries that are zero
    are actually slightly positive by an ad-hoc amount, in the interest
    of returning something more useful than NaN. This will cause a warning.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    if issubclass(A.dtype.type, np.integer):
        A = np.asarray(A, dtype=float)
    keep_it_real = np.isrealobj(A)
    try:
        if np.array_equal(A, np.triu(A)):
            A = _logm_force_nonsingular_triangular_matrix(A)
            if np.min(np.diag(A)) < 0:
                A = A.astype(complex)
            return _logm_triu(A)
        else:
            if keep_it_real:
                T, Z = schur(A)
                if not np.array_equal(T, np.triu(T)):
                    T, Z = rsf2csf(T, Z)
            else:
                T, Z = schur(A, output='complex')
            T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
            U = _logm_triu(T)
            ZH = np.conjugate(Z).T
            return Z.dot(U).dot(ZH)
    except (SqrtmError, LogmError):
        X = np.empty_like(A)
        X.fill(np.nan)
        return X