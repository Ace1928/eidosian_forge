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
def _get_nd_basic_indexing(self, key):
    """This function indexes ``self`` with a tuple of `slice` objects only."""
    key_nd = tuple((idx for idx in key if idx is not None))
    if len(key_nd) < self.ndim:
        raise RuntimeError('too few indices after normalization: expected `ndim` ({}) but got {}. This is a bug, please report it!'.format(self.ndim, len(key_nd)))
    if len(key_nd) > self.ndim:
        raise IndexError('too many indices ({}) for array with {} dimensions'.format(len(key_nd), self.ndim))
    slc_key, int_axes = self._basic_indexing_key_int_to_slice(key_nd)
    none_axes = [ax for ax in range(len(key)) if key[ax] is None]
    if none_axes:
        new_axes = self._new_axes_after_basic_indexing(none_axes, key)
    else:
        new_axes = []
    for ax in int_axes:
        if not -self.shape[ax] <= key_nd[ax] < self.shape[ax]:
            raise IndexError('index {} is out of bounds for axis {} with size {}'.format(key_nd[ax], ax, self.shape[ax]))
    begin, end, step = self._basic_indexing_key_to_begin_end_step(slc_key, self.shape, keep_none=False)
    if self._basic_indexing_slice_is_contiguous(slc_key, self.shape):
        flat_begin, flat_end = self._basic_indexing_contiguous_flat_begin_end(slc_key, self.shape)
        handle = NDArrayHandle()
        flat_self = self.reshape(-1)
        if _int64_enabled():
            check_call(_LIB.MXNDArraySlice64(flat_self.handle, ctypes.c_int64(flat_begin), ctypes.c_int64(flat_end), ctypes.byref(handle)))
        else:
            check_call(_LIB.MXNDArraySlice(flat_self.handle, ctypes.c_uint32(flat_begin), ctypes.c_uint32(flat_end), ctypes.byref(handle)))
        sliced_shape = self._basic_indexing_sliced_shape(slc_key, self.shape)
        sliced = NDArray(handle=handle, writable=self.writable).reshape(sliced_shape)
    else:
        begin, end, step = self._basic_indexing_key_to_begin_end_step(slc_key, self.shape, keep_none=True)
        sliced = op.slice(self, begin, end, step)
    final_shape = [sliced.shape[i] for i in range(sliced.ndim) if i not in int_axes]
    for ax in new_axes:
        final_shape.insert(ax, 1)
    if len(final_shape) == 0:
        final_shape = [1]
    return sliced.reshape(final_shape)