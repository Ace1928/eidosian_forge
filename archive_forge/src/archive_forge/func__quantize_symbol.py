import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
def _quantize_symbol(sym, ctx, excluded_symbols=None, excluded_operators=None, offline_params=None, quantized_dtype='int8', quantize_mode='smart', quantize_granularity='tensor-wise'):
    """Given a symbol object representing a neural network of data type FP32,
    quantize it into a INT8 network.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol.
    ctx : Context
        Defines the device that users want to run quantized symbol.
    excluded_symbols : list of strings
        A list of strings representing the names of the symbols that users want to excluding
        from being quantized.
    excluded_operators : list of strings
        A list of strings representing the names of the operators that users want to excluding
        from being quantized.
    offline_params : list of strs
        Names of the parameters that users want to quantize offline. It's always recommended to
        quantize parameters offline so that quantizing parameters during the inference can be
        avoided.
    quantized_dtype: str
        The quantized destination type for input data.
    quantize_mode: str
        The mode that quantization pass to apply.
    quantize_granularity: str
        The granularity of quantization, currently supports 'tensor-wise' and 'channel-wise'
        quantization. The default value is 'tensor-wise'.

    """
    num_excluded_symbols = 0
    if excluded_symbols is not None:
        assert isinstance(excluded_symbols, list)
        num_excluded_symbols = len(excluded_symbols)
    else:
        excluded_symbols = []
    num_excluded_ops = 0
    if excluded_operators is not None:
        assert isinstance(excluded_operators, list)
        num_excluded_ops = len(excluded_operators)
    else:
        excluded_operators = []
    num_offline = 0
    offline = []
    if offline_params is not None:
        num_offline = len(offline_params)
        for k in offline_params:
            offline.append(c_str(k))
    out = SymbolHandle()
    size = mx_uint()
    calib_str = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXQuantizeSymbol(sym.handle, ctypes.byref(out), ctypes.byref(ctypes.c_int(ctx.device_typeid)), mx_uint(num_excluded_symbols), c_str_array(excluded_symbols), mx_uint(num_excluded_ops), c_str_array(excluded_operators), mx_uint(num_offline), c_array(ctypes.c_char_p, offline), c_str(quantized_dtype), ctypes.c_bool(True), c_str(quantize_mode), c_str(quantize_granularity), ctypes.byref(size), ctypes.byref(calib_str)))
    calib_layer = []
    calib_layer = [py_str(calib_str[i]) for i in range(size.value)]
    return (Symbol(out), calib_layer)