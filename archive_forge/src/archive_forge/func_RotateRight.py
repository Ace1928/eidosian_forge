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
def RotateRight(a, b):
    """Return an expression representing `a` rotated to the right `b` times.

    >>> a, b = BitVecs('a b', 16)
    >>> RotateRight(a, b)
    RotateRight(a, b)
    >>> simplify(RotateRight(a, 0))
    a
    >>> simplify(RotateRight(a, 16))
    a
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_ext_rotate_right(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)