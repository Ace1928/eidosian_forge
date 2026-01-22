from array import array as native_array
import ctypes
import warnings
import numpy as _np
from ..autograd import is_recording
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray import indexing_key_expand_implicit_axes, get_indexing_dispatch_code,\
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle, c_array
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types, integer_types
from ..context import Context
from ..util import set_module, wrap_np_unary_func, wrap_np_binary_func
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi
from ..ndarray.ndarray import _storage_type, from_numpy
from .utils import _get_np_op
from .fallback import *  # pylint: disable=wildcard-import,unused-wildcard-import
from . import fallback
def _reshape_view(a, *shape):
    """Returns a **view** of this array with a new shape without altering any data.

    Parameters
    ----------
    shape : tuple of int, or n ints
        The new shape should not change the array size, namely
        ``np.prod(new_shape)`` should be equal to ``np.prod(a.shape)``.
        Some dimensions of the shape can take special value -1, which
        infers the dimension of the output shape by using the remainder of the
        input dimensions keeping the size of the new array same as that of the input array.
        At most one dimension of shape can be -1.

    Returns
    -------
    ndarray
        An array with desired shape that shares data with this array.
    """
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    handle = NDArrayHandle()
    check_call(_LIB.MXNDArrayReshape64(a.handle, len(shape), c_array(ctypes.c_int64, shape), False, ctypes.byref(handle)))
    return ndarray(handle=handle, writable=a.writable)