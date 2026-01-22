import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def _gather_type_ctx_info(args):
    """Analyze the elements inside the nested args object and find:
        - If there exists ndarray
        - If there exists symbol
        - All contexts appearing in args

    Parameters
    ----------
    args : list or NDArray or Symbol
        Could be a nested architecture.

    Returns
    -------
    has_symbol : bool
        Whether the elements in args contains symbols
    has_ndarray : bool
        Whether the elements in args contains ndarrays
    ctx_set : set of mxnet.context.Context
        Contains all possible contexts of the inner ndarrays in args. Can be empty if there is no
        ndarray inside args.
    first_ctx : mxnet.context.Context or None
        Context of the first appeared NDArray (for backward-compatibility)
    """
    if isinstance(args, NDArray):
        return (False, True, {args.ctx}, args.ctx)
    elif isinstance(args, Symbol):
        return (True, False, set(), None)
    elif isinstance(args, (list, tuple)):
        has_symbol = False
        has_ndarray = False
        ctx_set = set()
        first_ctx = None
        for ele in args:
            ele_has_sym, ele_has_nd, ele_ctx_set, ele_first_ctx = _gather_type_ctx_info(ele)
            has_symbol = has_symbol or ele_has_sym
            has_ndarray = has_ndarray or ele_has_nd
            if first_ctx is None and ele_first_ctx is not None:
                first_ctx = ele_first_ctx
            ctx_set = ctx_set | ele_ctx_set
            if has_symbol and has_ndarray:
                break
        return (has_symbol, has_ndarray, ctx_set, first_ctx)
    else:
        return (False, False, set(), None)