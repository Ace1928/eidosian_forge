import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
@classmethod
def _get_shape_and_sizes(cls, arrays, shape):
    sizes_shape = shape + tuple((a.size for a in arrays))
    return sizes_shape