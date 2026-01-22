import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def _correlate2d(in1, in2, mode, boundary, fillvalue, convolution=False):
    if not in1.ndim == in2.ndim == 2:
        raise ValueError('{} inputs must both be 2-D arrays'.format('convolve2d' if convolution else 'correlate2d'))
    _boundaries = {'fill': 'constant', 'pad': 'constant', 'wrap': 'wrap', 'circular': 'wrap', 'symm': 'reflect', 'symmetric': 'reflect'}
    boundary = _boundaries.get(boundary)
    if boundary is None:
        raise ValueError('Acceptable boundary flags are "fill" (or "pad"), "circular" (or "wrap"), and "symmetric" (or "symm").')
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    return _st_core._direct_correlate(in1, in2, mode, in1.dtype, convolution, boundary, fillvalue, not convolution)