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
def _check_parameter(func_name, order, mode):
    if order is None:
        warnings.warn(f'Currently the default order of {func_name} is 1. In a future release this may change to 3 to match scipy.ndimage ')
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')
    if mode not in ('constant', 'grid-constant', 'nearest', 'mirror', 'reflect', 'grid-mirror', 'wrap', 'grid-wrap', 'opencv', '_opencv_edge'):
        raise ValueError('boundary mode ({}) is not supported'.format(mode))