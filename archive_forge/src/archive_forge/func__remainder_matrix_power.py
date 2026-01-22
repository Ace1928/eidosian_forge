import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _remainder_matrix_power(A, t):
    """
    Compute the fractional power of a matrix, for fractions -1 < t < 1.

    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('input must be a square array')
    n, n = A.shape
    if np.array_equal(A, np.triu(A)):
        Z = None
        T = A
    elif np.isrealobj(A):
        T, Z = schur(A)
        if not np.array_equal(T, np.triu(T)):
            T, Z = rsf2csf(T, Z)
    else:
        T, Z = schur(A, output='complex')
    T_diag = np.diag(T)
    if np.count_nonzero(T_diag) != n:
        raise FractionalMatrixPowerError('cannot use inverse scaling and squaring to find the fractional matrix power of a singular matrix')
    if np.isrealobj(T) and np.min(T_diag) < 0:
        T = T.astype(complex)
    U = _remainder_matrix_power_triu(T, t)
    if Z is not None:
        ZH = np.conjugate(Z).T
        return Z.dot(U).dot(ZH)
    else:
        return U