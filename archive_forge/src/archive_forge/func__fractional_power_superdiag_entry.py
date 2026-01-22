import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _fractional_power_superdiag_entry(l1, l2, t12, p):
    """
    Compute a superdiagonal entry of a fractional matrix power.

    This is Eq. (5.6) in [1]_.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.
    p : float
        A fractional power.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the fractional matrix power.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if l1 == l2:
        f12 = t12 * p * l1 ** (p - 1)
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * (l2 ** p - l1 ** p) / (l2 - l1)
    else:
        z = (l2 - l1) / (l2 + l1)
        log_l1 = np.log(l1)
        log_l2 = np.log(l2)
        arctanh_z = np.arctanh(z)
        tmp_a = t12 * np.exp(p / 2 * (log_l2 + log_l1))
        tmp_u = _unwindk(log_l2 - log_l1)
        if tmp_u:
            tmp_b = p * (arctanh_z + np.pi * 1j * tmp_u)
        else:
            tmp_b = p * arctanh_z
        tmp_c = 2 * np.sinh(tmp_b) / (l2 - l1)
        f12 = tmp_a * tmp_c
    return f12