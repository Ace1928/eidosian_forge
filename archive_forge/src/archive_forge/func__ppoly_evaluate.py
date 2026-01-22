import math
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
def _ppoly_evaluate(c, x, xp, dx, extrapolate, out):
    """
    Evaluate a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bool
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    ascending = x[-1] >= x[0]
    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0], ascending))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    ppoly_kernel = _get_module_func(PPOLY_MODULE, 'eval_ppoly', c)
    ppoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (c, x, xp, intervals, dx, c_shape, c_strides, xp.shape[0], out))