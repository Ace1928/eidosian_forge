from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
def _infer_type_impl(self, partial, *args, **kwargs):
    """The actual implementation for calling type inference API."""
    if len(args) != 0 and len(kwargs) != 0:
        raise ValueError('Can only specify known argument                     types either by positional or kwargs way.')
    sdata = []
    if len(args) != 0:
        keys = c_array(ctypes.c_char_p, [])
        for s in args:
            if s is not None:
                s = _numpy.dtype(s).type
                if s not in _DTYPE_NP_TO_MX:
                    raise TypeError('Argument need to be one of ' + str(_DTYPE_NP_TO_MX))
                sdata.append(_DTYPE_NP_TO_MX[s])
            else:
                sdata.append(-1)
    else:
        str_keys = []
        for k, v in kwargs.items():
            v = _numpy.dtype(v).type
            if v in _DTYPE_NP_TO_MX:
                str_keys.append(k)
                sdata.append(_DTYPE_NP_TO_MX[v])
        keys = c_str_array(str_keys)
    arg_type_size = mx_uint()
    arg_type_data = ctypes.POINTER(ctypes.c_int)()
    out_type_size = mx_uint()
    out_type_data = ctypes.POINTER(ctypes.c_int)()
    aux_type_size = mx_uint()
    aux_type_data = ctypes.POINTER(ctypes.c_int)()
    complete = ctypes.c_int()
    if partial:
        infer_func = _LIB.MXSymbolInferTypePartial
    else:
        infer_func = _LIB.MXSymbolInferType
    check_call(infer_func(self.handle, mx_uint(len(sdata)), keys, c_array_buf(ctypes.c_int, array('i', sdata)), ctypes.byref(arg_type_size), ctypes.byref(arg_type_data), ctypes.byref(out_type_size), ctypes.byref(out_type_data), ctypes.byref(aux_type_size), ctypes.byref(aux_type_data), ctypes.byref(complete)))
    if complete.value != 0:
        arg_types = [_DTYPE_MX_TO_NP[arg_type_data[i]] for i in range(arg_type_size.value)]
        out_types = [_DTYPE_MX_TO_NP[out_type_data[i]] for i in range(out_type_size.value)]
        aux_types = [_DTYPE_MX_TO_NP[aux_type_data[i]] for i in range(aux_type_size.value)]
        return (arg_types, out_types, aux_types)
    else:
        return (None, None, None)