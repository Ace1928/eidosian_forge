from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
@staticmethod
def _drop_int_axes(indexed_shape, int_axes):
    """drop the axis of indexed_shape corresponding to int axes"""
    bcast_shape = []
    for i, size in enumerate(indexed_shape):
        if i not in int_axes:
            bcast_shape.append(size)
    if not bcast_shape:
        bcast_shape = [1]
    return tuple(bcast_shape)