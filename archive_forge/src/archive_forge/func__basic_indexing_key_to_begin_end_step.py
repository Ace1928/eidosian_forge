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
def _basic_indexing_key_to_begin_end_step(idcs, shape, keep_none=True):
    """Map a tuple of ``slice`` and ``None`` (ignored) to begin, end, step tuples."""
    idcs = [idx for idx in idcs if idx is not None]
    idcs = [idx if isinstance(idx, py_slice) else _int_to_slice(idx) for idx in idcs]
    if keep_none:
        sss_list = [(slc.start, slc.stop, slc.step) for slc, n in zip(idcs, shape)]
    else:
        sss_list = [slc.indices(n) for slc, n in zip(idcs, shape)]
    return tuple(zip(*sss_list))