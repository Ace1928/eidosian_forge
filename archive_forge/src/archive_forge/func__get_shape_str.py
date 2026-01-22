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
def _get_shape_str(args):

    def flatten(args):
        if not isinstance(args, (list, tuple)):
            return ([args], int(0))
        flat = []
        fmts = []
        for i in args:
            arg, fmt = flatten(i)
            flat.extend(arg)
            fmts.append(fmt)
        return (flat, fmts)

    def regroup(args, fmt):
        if isinstance(fmt, int):
            if fmt == 0:
                return (args[0], args[1:])
            return (args[:fmt], args[fmt:])
        ret = []
        for i in fmt:
            res, args = regroup(args, i)
            ret.append(res)
        return (ret, args)
    flat_args, fmts = flatten(args)
    flat_arg_shapes = [x.shape if isinstance(x, ndarray.NDArray) else x for x in flat_args]
    shapes = regroup(flat_arg_shapes, fmts)[0]
    if isinstance(shapes, list):
        shape_str = str(shapes)[1:-1]
    else:
        shape_str = str(shapes)
    return shape_str.replace('L', '')