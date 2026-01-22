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
def _get_index_range(start, stop, length, step=1):
    """Given start, stop, step and array length, return
    absolute values of start, stop, and step for generating index range.
    The returned values have been compensated by adding length if they
    are less than zero for all the cases but slice(None, None, -1).
    Note that the returned value of stop is not necessarily >= 0, since
    absolute stop is -1 in the case of slice(None, None, -1)."""
    if step == 0:
        raise ValueError('step size cannot be zero')
    if length < 0:
        raise ValueError('array length cannot be less than zero')
    if step is None:
        step = 1
    if start is None:
        if step > 0:
            start = 0
        else:
            start = length - 1
    elif start < 0:
        start += length
        if start < 0:
            start = 0
    elif start >= length:
        start = length
    if stop is None:
        if step > 0:
            stop = length
        else:
            stop = -1
    elif stop < 0:
        stop += length
        if stop < 0:
            stop = 0
    elif stop > length:
        stop = length
    return (start, stop, step)