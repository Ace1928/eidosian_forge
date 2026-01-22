import math
import cupy
from cupy.linalg import lstsq
from cupyx.scipy.ndimage import convolve1d
from ._arraytools import axis_slice
def _fit_edge(x, window_start, window_stop, interp_start, interp_stop, axis, polyorder, deriv, delta, y):
    """
    Given an N-d array `x` and the specification of a slice of `x` from
    `window_start` to `window_stop` along `axis`, create an interpolating
    polynomial of each 1-D slice, and evaluate that polynomial in the slice
    from `interp_start` to `interp_stop`. Put the result into the
    corresponding slice of `y`.
    """
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)
    poly_coeffs = cupy.polyfit(cupy.arange(0, window_stop - window_start), xx_edge, polyorder)
    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)
    i = cupy.arange(interp_start - window_start, interp_stop - window_start)
    values = _polyval(poly_coeffs, i.reshape(-1, 1)) / delta ** deriv
    shp = list(y.shape)
    shp[0], shp[axis] = (shp[axis], shp[0])
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    if swapped:
        values = values.swapaxes(0, axis)
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values