import functools
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import atleast_nd, array_namespace
def _compute_absolute_step(rel_step, x0, f0, method):
    """
    Computes an absolute step from a relative step for finite difference
    calculation.

    Parameters
    ----------
    rel_step: None or array-like
        Relative step for the finite difference calculation
    x0 : np.ndarray
        Parameter vector
    f0 : np.ndarray or scalar
    method : {'2-point', '3-point', 'cs'}

    Returns
    -------
    h : float
        The absolute step size

    Notes
    -----
    `h` will always be np.float64. However, if `x0` or `f0` are
    smaller floating point dtypes (e.g. np.float32), then the absolute
    step size will be calculated from the smallest floating point size.
    """
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1
    rstep = _eps_for_method(x0.dtype, f0.dtype, method)
    if rel_step is None:
        abs_step = rstep * sign_x0 * np.maximum(1.0, np.abs(x0))
    else:
        abs_step = rel_step * sign_x0 * np.abs(x0)
        dx = x0 + abs_step - x0
        abs_step = np.where(dx == 0, rstep * sign_x0 * np.maximum(1.0, np.abs(x0)), abs_step)
    return abs_step