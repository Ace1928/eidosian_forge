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
def _is_advanced_index(idx):
    """Return whether ``idx`` is an advanced index (array-like or integer).

    Note that in contrast to basic indexing, integers are considered advanced
    indices in the context of advanced indexing as they participate in
    broadcasting.
    """
    if isinstance(idx, (NDArray, np.ndarray, integer_types, list, tuple)):
        return True
    elif isinstance(idx, py_slice) or idx is None:
        return False
    elif isinstance(idx, range):
        return True
    else:
        raise RuntimeError('illegal index type {}'.format(type(idx)))