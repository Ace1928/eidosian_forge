import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def _fftconv_faster(x, h, mode):
    """
    .. seealso:: :func: `scipy.signal._signaltools._fftconv_faster`

    """
    return True