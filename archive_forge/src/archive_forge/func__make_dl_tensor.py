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
def _make_dl_tensor(array):
    if str(array.dtype) not in DLDataType.TYPE_MAP:
        raise ValueError(str(array.dtype) + ' is not supported.')
    dl_tensor = DLTensor()
    dl_tensor.data = array.ctypes.data_as(ctypes.c_void_p)
    dl_tensor.ctx = DLContext(1, 0)
    dl_tensor.ndim = array.ndim
    dl_tensor.dtype = DLDataType.TYPE_MAP[str(array.dtype)]
    dl_tensor.shape = array.ctypes.shape_as(ctypes.c_int64)
    dl_tensor.strides = None
    dl_tensor.byte_offset = 0
    return dl_tensor