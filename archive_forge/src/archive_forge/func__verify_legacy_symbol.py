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
def _verify_legacy_symbol(op_name, func_name, sym):
    """Verify if the sym is a legacy symbol.

    Parameters
    ----------
    op_name : str
        Operator full name registered in backend.
    func_name : str
        Operator name exposed to users. This is usually the name by stripping off
        the prefix of the full operator names registered in backend.
    sym : symbol to be verified
    """
    from .numpy._symbol import _Symbol as np_symbol
    if isinstance(sym, np_symbol):
        raise TypeError('Operator `{}` registered in backend is known as `{}` in Python. This is a legacy operator which can only accept legacy ndarrays, while received an MXNet numpy ndarray. Please call `as_nd_ndarray()` upon the numpy ndarray to convert it to a legacy ndarray, and then feed the converted array to this operator.'.format(op_name, func_name))