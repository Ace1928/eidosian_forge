import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def _choose_conv_method(in1, in2, mode):
    if in1.ndim != 1 or in2.ndim != 1:
        raise NotImplementedError('Only 1d inputs are supported currently')
    if in1.dtype.kind in 'bui' or in2.dtype.kind in 'bui':
        return 'direct'
    if _fftconv_faster(in1, in2, mode):
        return 'fft'
    return 'direct'