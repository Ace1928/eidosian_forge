import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _origins_to_offsets(origins, w_shape):
    return tuple((x // 2 + o for x, o in zip(w_shape, origins)))