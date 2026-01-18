import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def visit_vars_inner(node, callback, cbdata):
    if isinstance(node, ir.Var):
        return callback(node, cbdata)
    elif isinstance(node, list):
        return [visit_vars_inner(n, callback, cbdata) for n in node]
    elif isinstance(node, tuple):
        return tuple([visit_vars_inner(n, callback, cbdata) for n in node])
    elif isinstance(node, ir.Expr):
        for arg in node._kws.keys():
            node._kws[arg] = visit_vars_inner(node._kws[arg], callback, cbdata)
    elif isinstance(node, ir.Yield):
        node.value = visit_vars_inner(node.value, callback, cbdata)
    return node