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
def _new_from_shared_mem(shared_pid, shared_id, shape, dtype):
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateFromSharedMemEx(ctypes.c_int(shared_pid), ctypes.c_int(shared_id), c_array(mx_int, shape), mx_int(len(shape)), ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])), ctypes.byref(hdl)))
    return hdl