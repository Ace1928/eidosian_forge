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
def _get_nd_advanced_indexing(self, key):
    """Get item when key is a tuple of any objects of the following types:
        NDArray, np.ndarray, list, tuple, slice, and integer."""
    slc_key, new_axes = self._get_index_nd(key)
    sliced = op.gather_nd(self, slc_key)
    if new_axes:
        final_shape = [sliced.shape[i] for i in range(sliced.ndim)]
        for ax in new_axes:
            final_shape.insert(ax, 1)
        return sliced.reshape(final_shape)
    else:
        return sliced