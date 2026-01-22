import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _check_conv_inputs(in1, in2, mode, convolution=True):
    if in1.ndim == in2.ndim == 0:
        return in1 * (in2 if convolution else in2.conj())
    if in1.ndim != in2.ndim:
        raise ValueError('in1 and in2 should have the same dimensionality')
    if in1.size == 0 or in2.size == 0:
        return cupy.array([], dtype=in1.dtype)
    if mode not in ('full', 'same', 'valid'):
        raise ValueError('acceptable modes are "valid", "same", or "full"')
    return None