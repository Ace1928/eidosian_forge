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
def _broadcast_advanced_indices(arrays, block_axes):
    """Broadcast arrays according to position in the sequence.

        Here, "according to position" means that an array of dimension 1
        (which is the case for all except ``block_axes``) will have shape
        ``(1, ..., 1, N, 1, ..., 1)``, where ``N`` is the length, and the
        position of ``N`` in the shape is the same as the position of the
        array in the ``arrays`` sequence, plus extra dimensions of the
        advanced block if it is left of the array.

        The arrays at ``block_axes`` are the advanced indices. They are assumed to
        be ready for mutual broadcasting to produce the advanced indexing block.
        It is further assumed that the numbers in ``block_axes`` are consecutive.

        The return value is a tuple containing the arrays with broadcast shapes.
        """
    block_shape = _broadcast_shapes([arrays[ax] for ax in block_axes])
    ndim_blk = len(block_shape)
    ndim_blk_delta = ndim_blk - len(block_axes)
    ndim_lead = block_axes[0]
    ndim_trail = len(arrays) - (block_axes[-1] + 1)
    bcast_shape = tuple((arrays[ax].shape[0] for ax in range(ndim_lead))) + block_shape + tuple((arrays[ax].shape[0] for ax in range(block_axes[-1] + 1, len(arrays))))
    bcast_arrays = [None] * len(arrays)
    for ax in block_axes:
        arr = arrays[ax].broadcast_to(block_shape)
        shp = (1,) * ndim_lead + block_shape + (1,) * ndim_trail
        bcast_arrays[ax] = arr.reshape(shp).broadcast_to(bcast_shape)
    for ax in set(range(len(arrays))) - set(block_axes):
        shp = [1] * len(bcast_shape)
        if ax < ndim_lead:
            shp[ax] = arrays[ax].shape[0]
        else:
            shp[ax + ndim_blk_delta] = arrays[ax].shape[0]
        bcast_arrays[ax] = arrays[ax].reshape(shp).broadcast_to(bcast_shape)
    return tuple(bcast_arrays)