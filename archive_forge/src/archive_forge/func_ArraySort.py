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
def ArraySort(*sig):
    """Return the Z3 array sort with the given domain and range sorts.

    >>> A = ArraySort(IntSort(), BoolSort())
    >>> A
    Array(Int, Bool)
    >>> A.domain()
    Int
    >>> A.range()
    Bool
    >>> AA = ArraySort(IntSort(), A)
    >>> AA
    Array(Int, Array(Int, Bool))
    """
    sig = _get_args(sig)
    if z3_debug():
        _z3_assert(len(sig) > 1, 'At least two arguments expected')
    arity = len(sig) - 1
    r = sig[arity]
    d = sig[0]
    if z3_debug():
        for s in sig:
            _z3_assert(is_sort(s), 'Z3 sort expected')
            _z3_assert(s.ctx == r.ctx, 'Context mismatch')
    ctx = d.ctx
    if len(sig) == 2:
        return ArraySortRef(Z3_mk_array_sort(ctx.ref(), d.ast, r.ast), ctx)
    dom = (Sort * arity)()
    for i in range(arity):
        dom[i] = sig[i].ast
    return ArraySortRef(Z3_mk_array_sort_n(ctx.ref(), arity, dom, r.ast), ctx)