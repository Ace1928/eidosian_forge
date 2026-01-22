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
def asscalar(self):
    """Returns a scalar whose value is copied from this array.

        This function is equivalent to ``self.asnumpy()[0]``. This NDArray must have shape (1,).

        Examples
        --------
        >>> x = mx.nd.ones((1,), dtype='int32')
        >>> x.asscalar()
        1
        >>> type(x.asscalar())
        <type 'numpy.int32'>
        """
    if self.size != 1:
        raise ValueError('The current array is not a scalar')
    if self.ndim == 1:
        return self.asnumpy()[0]
    else:
        return self.asnumpy()[()]