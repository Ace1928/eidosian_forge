from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def FreshFunction(*sig):
    """Create a new fresh Z3 uninterpreted function with the given sorts.
    """
    sig = _get_args(sig)
    if z3_debug():
        _z3_assert(len(sig) > 0, 'At least two arguments expected')
    arity = len(sig) - 1
    rng = sig[arity]
    if z3_debug():
        _z3_assert(is_sort(rng), 'Z3 sort expected')
    dom = (z3.Sort * arity)()
    for i in range(arity):
        if z3_debug():
            _z3_assert(is_sort(sig[i]), 'Z3 sort expected')
        dom[i] = sig[i].ast
    ctx = rng.ctx
    return FuncDeclRef(Z3_mk_fresh_func_decl(ctx.ref(), 'f', arity, dom, rng.ast), ctx)