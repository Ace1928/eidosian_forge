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
def _new_alloc_handle(stype, shape, ctx, delay_alloc, dtype, aux_types, aux_shapes=None):
    """Return a new handle with specified storage type, shape, dtype and context.

    Empty handle is only used to hold results

    Returns
    -------
    handle
        A new empty ndarray handle
    """
    hdl = NDArrayHandle()
    for aux_t in aux_types:
        if np.dtype(aux_t) != np.dtype('int64'):
            raise NotImplementedError('only int64 is supported for aux types')
    aux_type_ids = [int(_DTYPE_NP_TO_MX[np.dtype(aux_t).type]) for aux_t in aux_types]
    aux_shapes = [(0,) for aux_t in aux_types] if aux_shapes is None else aux_shapes
    aux_shape_lens = [len(aux_shape) for aux_shape in aux_shapes]
    aux_shapes = py_sum(aux_shapes, ())
    num_aux = ctypes.c_uint(len(aux_types))
    if _int64_enabled():
        check_call(_LIB.MXNDArrayCreateSparseEx64(ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[stype])), c_array_buf(ctypes.c_int64, native_array('q', shape)), ctypes.c_int(len(shape)), ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id), ctypes.c_int(int(delay_alloc)), ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])), num_aux, c_array_buf(ctypes.c_int, native_array('i', aux_type_ids)), c_array_buf(ctypes.c_int, native_array('i', aux_shape_lens)), c_array_buf(ctypes.c_int64, native_array('q', aux_shapes)), ctypes.byref(hdl)))
    else:
        check_call(_LIB.MXNDArrayCreateSparseEx(ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[stype])), c_array_buf(ctypes.c_uint, native_array('I', shape)), ctypes.c_uint(len(shape)), ctypes.c_int(ctx.device_typeid), ctypes.c_int(ctx.device_id), ctypes.c_int(int(delay_alloc)), ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])), num_aux, c_array_buf(ctypes.c_int, native_array('i', aux_type_ids)), c_array_buf(ctypes.c_uint, native_array('I', aux_shape_lens)), c_array_buf(ctypes.c_uint, native_array('I', aux_shapes)), ctypes.byref(hdl)))
    return hdl