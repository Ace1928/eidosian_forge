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
def transfer_scope(block, scope):
    """Transfer the ir.Block to use the given ir.Scope.
    """
    old_scope = block.scope
    if old_scope is scope:
        return block
    for var in old_scope.localvars._con.values():
        if var.name not in scope.localvars:
            scope.localvars.define(var.name, var)
    block.scope = scope
    return block