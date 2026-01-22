from array import array
import ctypes
import warnings
from ..ndarray import NDArray
from ..base import _LIB, c_str_array, c_handle_array, c_array, c_array_buf, c_str
from ..base import check_call, string_types
from ..base import KVStoreHandle
from ..profiler import set_kvstore_handle
def _ctype_dict(param_dict):
    """Returns ctype arrays for keys and values(converted to strings) in a dictionary"""
    assert isinstance(param_dict, dict), 'unexpected type for param_dict: ' + str(type(param_dict))
    c_keys = c_array(ctypes.c_char_p, [c_str(k) for k in param_dict.keys()])
    c_vals = c_array(ctypes.c_char_p, [c_str(str(v)) for v in param_dict.values()])
    return (c_keys, c_vals)