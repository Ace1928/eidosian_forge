import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _inverse_squaring_helper(T0, theta):
    """
    A helper function for inverse scaling and squaring for Pade approximation.

    Parameters
    ----------
    T0 : (N, N) array_like upper triangular
        Matrix involved in inverse scaling and squaring.
    theta : indexable
        The values theta[1] .. theta[7] must be available.
        They represent bounds related to Pade approximation, and they depend
        on the matrix function which is being computed.
        For example, different values of theta are required for
        matrix logarithm than for fractional matrix power.

    Returns
    -------
    R : (N, N) array_like upper triangular
        Composition of zero or more matrix square roots of T0, minus I.
    s : non-negative integer
        Number of square roots taken.
    m : positive integer
        The degree of the Pade approximation.

    Notes
    -----
    This subroutine appears as a chunk of lines within
    a couple of published algorithms; for example it appears
    as lines 4--35 in algorithm (3.1) of [1]_, and
    as lines 3--34 in algorithm (4.1) of [2]_.
    The instances of 'goto line 38' in algorithm (3.1) of [1]_
    probably mean 'goto line 36' and have been interpreted accordingly.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    if len(T0.shape) != 2 or T0.shape[0] != T0.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T0.shape
    T = T0
    s0 = 0
    tmp_diag = np.diag(T)
    if np.count_nonzero(tmp_diag) != n:
        raise Exception('Diagonal entries of T must be nonzero')
    while np.max(np.absolute(tmp_diag - 1)) > theta[7]:
        tmp_diag = np.sqrt(tmp_diag)
        s0 += 1
    for i in range(s0):
        T = _sqrtm_triu(T)
    s = s0
    k = 0
    d2 = _onenormest_m1_power(T, 2) ** (1 / 2)
    d3 = _onenormest_m1_power(T, 3) ** (1 / 3)
    a2 = max(d2, d3)
    m = None
    for i in (1, 2):
        if a2 <= theta[i]:
            m = i
            break
    while m is None:
        if s > s0:
            d3 = _onenormest_m1_power(T, 3) ** (1 / 3)
        d4 = _onenormest_m1_power(T, 4) ** (1 / 4)
        a3 = max(d3, d4)
        if a3 <= theta[7]:
            j1 = min((i for i in (3, 4, 5, 6, 7) if a3 <= theta[i]))
            if j1 <= 6:
                m = j1
                break
            elif a3 / 2 <= theta[5] and k < 2:
                k += 1
                T = _sqrtm_triu(T)
                s += 1
                continue
        d5 = _onenormest_m1_power(T, 5) ** (1 / 5)
        a4 = max(d4, d5)
        eta = min(a3, a4)
        for i in (6, 7):
            if eta <= theta[i]:
                m = i
                break
        if m is not None:
            break
        T = _sqrtm_triu(T)
        s += 1
    R = T - np.identity(n)
    has_principal_branch = all((x.real > 0 or x.imag != 0 for x in np.diag(T0)))
    if has_principal_branch:
        for j in range(n):
            a = T0[j, j]
            r = _briggs_helper_function(a, s)
            R[j, j] = r
        p = np.exp2(-s)
        for j in range(n - 1):
            l1 = T0[j, j]
            l2 = T0[j + 1, j + 1]
            t12 = T0[j, j + 1]
            f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
            R[j, j + 1] = f12
    if not np.array_equal(R, np.triu(R)):
        raise Exception('R is not upper triangular')
    return (R, s, m)