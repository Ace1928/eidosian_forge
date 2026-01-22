import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
def _prepare_default_dtype(src_array, dtype):
    """Prepare the value of dtype if `dtype` is None. If `src_array` is an NDArray, numpy.ndarray
    or scipy.sparse.csr.csr_matrix, return src_array.dtype. float32 is returned otherwise."""
    if dtype is None:
        if isinstance(src_array, (NDArray, np.ndarray)):
            dtype = src_array.dtype
        elif spsp and isinstance(src_array, spsp.csr.csr_matrix):
            dtype = src_array.dtype
        else:
            dtype = mx_real_t
    return dtype