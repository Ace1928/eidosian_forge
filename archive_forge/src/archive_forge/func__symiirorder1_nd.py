import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def _symiirorder1_nd(input, c0, z1, precision=-1.0, axis=-1):
    axis = _normalize_axis_index(axis, input.ndim)
    input_shape = input.shape
    input_ndim = input.ndim
    if input.ndim > 1:
        input, input_shape = collapse_2d(input, axis)
    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        if input.dtype is cupy.dtype(cupy.float64):
            precision = 1e-06
        elif input.dtype is cupy.dtype(cupy.float32):
            precision = 0.001
        else:
            precision = 10 ** (-cupy.finfo(input.dtype).iexp)
    precision *= precision
    pos = cupy.arange(1, input_shape[-1] + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos
    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input, axis=-1) + axis_slice(input, 0, 1, axis=-1)
    all_valid = diff <= precision
    zi = _find_initial_cond(all_valid, cum_poly, input_shape[-1])
    if cupy.any(cupy.isnan(zi)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    zi_shape = (1, 4)
    if input_ndim > 1:
        zi_shape = (1, input.shape[0], 4)
    all_zi = cupy.zeros(zi_shape, dtype=input.dtype)
    all_zi = axis_assign(all_zi, zi, 3, 4)
    coef = cupy.r_[1, 0, 0, 1, -z1, 0]
    coef = cupy.atleast_2d(coef)
    y1, _ = apply_iir_sos(axis_slice(input, 1), coef, zi=all_zi, dtype=input.dtype, apply_fir=False)
    y1 = cupy.c_[zi, y1]
    zi = -c0 / (z1 - 1.0) * axis_slice(y1, -1)
    all_zi = axis_assign(all_zi, zi, 3, 4)
    coef = cupy.r_[c0, 0, 0, 1, -z1, 0]
    coef = cupy.atleast_2d(coef)
    out, _ = apply_iir_sos(axis_slice(y1, -2, step=-1), coef, zi=all_zi, dtype=input.dtype)
    if input_ndim > 1:
        out = cupy.c_[axis_reverse(out), zi]
    else:
        out = cupy.r_[axis_reverse(out), zi]
    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out