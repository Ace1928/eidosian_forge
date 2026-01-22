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
def _load_cached_graphs(blk, index, structure):
    name = type(blk).__name__.lower()
    mdl = structure[name + str(index[0])]
    blk._name = mdl['orig_name']
    if isinstance(blk, HybridBlock):
        if mdl['hybridized']:
            blk._in_format = mdl['in_format']
            blk._out_format = mdl['out_format']
            out = load_json(mdl['symbol'])
            syms = []
            for inp in mdl['inputs']:
                syms.append(load_json(inp))
            blk._cached_graph = (syms, out)
            blk._active = True
    pnames = list(blk.params.keys())
    for p in pnames:
        param = blk.params._params[p]
        new_name = blk.name + '_' + p[len(blk.params._prefix):]
        blk.params._params.pop(p)
        blk.params._params[new_name] = param
    for ch_name, child in blk._children.items():
        index[0] += 1
        _load_cached_graphs(child, index, mdl)
    ch_names = list(blk._children.keys())
    children = mdl['children']
    for ch_name in ch_names:
        child = blk._children[ch_name]
        blk._children.pop(ch_name)
        orig_name = children[child.name]
        blk._children[orig_name] = child