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
def BitVecVal(val, bv, ctx=None):
    """Return a bit-vector value with the given number of bits. If `ctx=None`, then the global context is used.

    >>> v = BitVecVal(10, 32)
    >>> v
    10
    >>> print("0x%.8x" % v.as_long())
    0x0000000a
    """
    if is_bv_sort(bv):
        ctx = bv.ctx
        return BitVecNumRef(Z3_mk_numeral(ctx.ref(), _to_int_str(val), bv.ast), ctx)
    else:
        ctx = _get_ctx(ctx)
        return BitVecNumRef(Z3_mk_numeral(ctx.ref(), _to_int_str(val), BitVecSort(bv, ctx).ast), ctx)