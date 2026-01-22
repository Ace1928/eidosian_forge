import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _get_positions(arrays, position_arrays, arg_func):
    """Concatenated positions from applying arg_func to arrays.

    arg_func should be cupy.argmin or cupy.argmax
    """
    return cupy.concatenate([pos[arg_func(a, keepdims=True)] if a.size != 0 else cupy.asarray([0], dtype=int) for pos, a in zip(position_arrays, arrays)])