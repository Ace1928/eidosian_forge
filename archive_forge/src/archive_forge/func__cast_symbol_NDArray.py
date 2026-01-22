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
def _cast_symbol_NDArray(s, dtype):
    float_types_gpu = (np.float16, np.float32)
    float_types_cpu = (bfloat16, np.float32)
    if isinstance(s, Symbol):
        return symbol.amp_cast(s, dtype=dtype)
    elif isinstance(s, NDArray):
        if s.dtype != dtype and s.dtype in float_types_gpu and (s.context.device_type != 'cpu'):
            return ndarray.amp_cast(s, dtype=dtype)
        elif s.dtype != dtype and s.dtype in float_types_cpu and (s.context.device_type == 'cpu'):
            return ndarray.amp_cast(s, dtype=dtype)
        else:
            return s
    else:
        return s