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
class DLDataType(ctypes.Structure):
    _fields_ = [('type_code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]
    TYPE_MAP = {'int32': (0, 32, 1), 'int64': (0, 64, 1), 'bool': (1, 1, 1), 'uint8': (1, 8, 1), 'uint32': (1, 32, 1), 'uint64': (1, 64, 1), 'float16': (2, 16, 1), 'float32': (2, 32, 1), 'float64': (2, 64, 1)}