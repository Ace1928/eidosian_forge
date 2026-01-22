import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
def _unwindk(z):
    """
    Compute the scalar unwinding number.

    Uses Eq. (5.3) in [1]_, and should be equal to (z - log(exp(z)) / (2 pi i).
    Note that this definition differs in sign from the original definition
    in equations (5, 6) in [2]_.  The sign convention is justified in [3]_.

    Parameters
    ----------
    z : complex
        A complex number.

    Returns
    -------
    unwinding_number : integer
        The scalar unwinding number of z.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    .. [2] Robert M. Corless and David J. Jeffrey,
           "The unwinding number." Newsletter ACM SIGSAM Bulletin
           Volume 30, Issue 2, June 1996, Pages 28-35.

    .. [3] Russell Bradford and Robert M. Corless and James H. Davenport and
           David J. Jeffrey and Stephen M. Watt,
           "Reasoning about the elementary functions of complex analysis"
           Annals of Mathematics and Artificial Intelligence,
           36: 303-318, 2002.

    """
    return int(np.ceil((z.imag - np.pi) / (2 * np.pi)))