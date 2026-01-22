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
def PropagateFunction(name, *sig):
    """Create a function that gets tracked by user propagator.
       Every term headed by this function symbol is tracked.
       If a term is fixed and the fixed callback is registered a
       callback is invoked that the term headed by this function is fixed.
    """
    sig = _get_args(sig)
    if z3_debug():
        _z3_assert(len(sig) > 0, 'At least two arguments expected')
    arity = len(sig) - 1
    rng = sig[arity]
    if z3_debug():
        _z3_assert(is_sort(rng), 'Z3 sort expected')
    dom = (Sort * arity)()
    for i in range(arity):
        if z3_debug():
            _z3_assert(is_sort(sig[i]), 'Z3 sort expected')
        dom[i] = sig[i].ast
    ctx = rng.ctx
    return FuncDeclRef(Z3_solver_propagate_declare(ctx.ref(), to_symbol(name, ctx), arity, dom, rng.ast), ctx)