import math
import warnings
import cupy
import numpy
from cupy import _core
from cupy._core import internal
from cupy.cuda import runtime
from cupyx import _texture
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core
def _prepad_for_spline_filter(input, mode, cval):
    if mode in ['nearest', 'grid-constant']:
        npad = 12
        if mode == 'grid-constant':
            kwargs = dict(mode='constant', constant_values=cval)
        else:
            kwargs = dict(mode='edge')
        padded = cupy.pad(input, npad, **kwargs)
    else:
        npad = 0
        padded = input
    return (padded, npad)