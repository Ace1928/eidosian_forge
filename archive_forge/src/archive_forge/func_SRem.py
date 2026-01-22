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
def SRem(a, b):
    """Create the Z3 expression signed remainder.

    Use the operator % for signed modulus, and URem() for unsigned remainder.

    >>> x = BitVec('x', 32)
    >>> y = BitVec('y', 32)
    >>> SRem(x, y)
    SRem(x, y)
    >>> SRem(x, y).sort()
    BitVec(32)
    >>> (x % y).sexpr()
    '(bvsmod x y)'
    >>> SRem(x, y).sexpr()
    '(bvsrem x y)'
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_bvsrem(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)