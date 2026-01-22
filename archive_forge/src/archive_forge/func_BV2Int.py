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
def BV2Int(a, is_signed=False):
    """Return the Z3 expression BV2Int(a).

    >>> b = BitVec('b', 3)
    >>> BV2Int(b).sort()
    Int
    >>> x = Int('x')
    >>> x > BV2Int(b)
    x > BV2Int(b)
    >>> x > BV2Int(b, is_signed=False)
    x > BV2Int(b)
    >>> x > BV2Int(b, is_signed=True)
    x > If(b < 0, BV2Int(b) - 8, BV2Int(b))
    >>> solve(x > BV2Int(b), b == 1, x < 3)
    [x = 2, b = 1]
    """
    if z3_debug():
        _z3_assert(is_bv(a), 'First argument must be a Z3 bit-vector expression')
    ctx = a.ctx
    return ArithRef(Z3_mk_bv2int(ctx.ref(), a.as_ast(), is_signed), ctx)