import os as _os
import ctypes
import numpy as _np
from . import _internal
from ._internal import SymbolBase, _symbol_creator
from ..attribute import AttrScope
from ..base import mx_uint, check_call, _LIB, py_str
from ..symbol_doc import _build_doc
from ..base import _Null, _init_op_module, _is_np_op, _output_is_list
from ..name import NameManager
from .contrib import adamw_update, mp_adamw_update
from ._internal import _adamw_update, _mp_adamw_update
def _make_symbol_function(handle, name, func_name):
    """Create a symbol function by handle and function name."""
    code, doc_str = _generate_symbol_function_code(handle, name, func_name)
    local = {}
    exec(code, None, local)
    symbol_function = local[func_name]
    symbol_function.__name__ = func_name
    symbol_function.__doc__ = doc_str
    symbol_function.__module__ = 'mxnet.symbol'
    return symbol_function