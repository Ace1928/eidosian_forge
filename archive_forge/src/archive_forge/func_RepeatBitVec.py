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
def RepeatBitVec(n, a):
    """Return an expression representing `n` copies of `a`.

    >>> x = BitVec('x', 8)
    >>> n = RepeatBitVec(4, x)
    >>> n
    RepeatBitVec(4, x)
    >>> n.size()
    32
    >>> v0 = BitVecVal(10, 4)
    >>> print("%.x" % v0.as_long())
    a
    >>> v = simplify(RepeatBitVec(4, v0))
    >>> v.size()
    16
    >>> print("%.x" % v.as_long())
    aaaa
    """
    if z3_debug():
        _z3_assert(_is_int(n), 'First argument must be an integer')
        _z3_assert(is_bv(a), 'Second argument must be a Z3 bit-vector expression')
    return BitVecRef(Z3_mk_repeat(a.ctx_ref(), n, a.as_ast()), a.ctx)