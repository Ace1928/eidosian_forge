import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
def _get_kvstore_server_command_type(command):
    command_types = {'kController': 0, 'kSetMultiPrecision': 1, 'kStopServer': 2, 'kSyncMode': 3, 'kSetGradientCompression': 4, 'kSetProfilerParams': 5}
    assert command in command_types, 'Unknown command type to send to server'
    return command_types[command]