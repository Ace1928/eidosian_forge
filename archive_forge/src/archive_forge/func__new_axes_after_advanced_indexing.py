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
def _new_axes_after_advanced_indexing(key, adv_axs, bcast_adv_ndim, adv_are_adjacent):
    """
        Return indices of ``axes`` after slicing with ``key_nd``.

        This function is used to calculate the positions where new axes should
        end up after indexing, taking into account the removal of axes by
        integer indexing.

        The ``key`` sequence should be the exapanded key including slices, array like objects,
        integer types and ``None``.
        ``adv_axes`` is the sequence of indices of advanced axes.
        ``bcast_adv_ndim`` is the number of dimensions of advanced indexing subspace.
        ``adv_are_adjacent`` is a boolean value. Value being True means all advanced indicies are adjacent.

        Note: integer indices are also considered advanced indices here.
        """
    new_axes = [ax for ax in range(len(key)) if key[ax] is None]
    adv_axs_set = set(adv_axs)
    if not adv_are_adjacent:
        steps = [bcast_adv_ndim] + [0 if ax in adv_axs_set else 1 for ax in range(len(key))]
    else:
        steps = [0] + [0 if ax in adv_axs_set else 1 for ax in range(len(key))]
    cum_steps = np.cumsum(steps)
    axes_after = tuple(cum_steps[new_axes])
    return axes_after