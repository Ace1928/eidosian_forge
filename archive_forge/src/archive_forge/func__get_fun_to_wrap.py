from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def _get_fun_to_wrap(name, module, submodule_dict):
    module_internal = getattr(module, '_internal')
    prefix = base._get_op_name_prefix(name)
    if len(prefix) > 0:
        if prefix != '_random_' or name.endswith('_like'):
            func_name = name[len(prefix):]
            cur_module = submodule_dict[prefix]
        else:
            func_name = name
            cur_module = module_internal
    elif name.startswith('_'):
        func_name = name
        cur_module = module_internal
    else:
        func_name = name
        cur_module = module
    return (func_name, cur_module)