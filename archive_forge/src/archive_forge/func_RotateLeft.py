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
def RotateLeft(a, b):
    """Return an expression representing `a` rotated to the left `b` times.

    >>> a, b = BitVecs('a b', 16)
    >>> RotateLeft(a, b)
    RotateLeft(a, b)
    >>> simplify(RotateLeft(a, 0))
    a
    >>> simplify(RotateLeft(a, 16))
    a
    """
    _check_bv_args(a, b)
    a, b = _coerce_exprs(a, b)
    return BitVecRef(Z3_mk_ext_rotate_left(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)