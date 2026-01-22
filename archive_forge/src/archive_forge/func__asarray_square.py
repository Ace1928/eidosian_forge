from itertools import product
import numpy as np
from numpy import (dot, diag, prod, logical_not, ravel, transpose,
from numpy.lib.scimath import sqrt as csqrt
from scipy.linalg import LinAlgError, bandwidth
from ._misc import norm
from ._basic import solve, inv
from ._decomp_svd import svd
from ._decomp_schur import schur, rsf2csf
from ._expm_frechet import expm_frechet, expm_cond
from ._matfuncs_sqrtm import sqrtm
from ._matfuncs_expm import pick_pade_structure, pade_UV_calc
from numpy import single  # noqa: F401
def _asarray_square(A):
    """
    Wraps asarray with the extra requirement that the input be a square matrix.

    The motivation is that the matfuncs module has real functions that have
    been lifted to square matrix functions.

    Parameters
    ----------
    A : array_like
        A square matrix.

    Returns
    -------
    out : ndarray
        An ndarray copy or view or other representation of A.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square array_like input')
    return A