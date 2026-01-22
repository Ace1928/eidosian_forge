import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
def _set_tvm_op_config(x):
    """ctypes implementation of populating the config singleton"""
    check_call(_LIB.MXLoadTVMConfig(c_config_spaces(x)))
    return x