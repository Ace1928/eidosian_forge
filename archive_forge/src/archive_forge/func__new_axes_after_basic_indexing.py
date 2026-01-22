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
def _new_axes_after_basic_indexing(axes, key):
    """Return indices of ``axes`` after slicing with ``key``.

        This function is used to calculate the positions where new axes should
        end up after indexing, taking into account the removal of axes by
        integer indexing.

        The ``key`` sequence should be the exapanded key including slices, integer types
        and ``None``.
        """
    steps = [0] + [0 if isinstance(idx, integer_types) else 1 for idx in key]
    cum_steps = np.cumsum(steps)
    axes_after = tuple(cum_steps[axes])
    return axes_after