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
def ToReal(a):
    """ Return the Z3 expression ToReal(a).

    >>> x = Int('x')
    >>> x.sort()
    Int
    >>> n = ToReal(x)
    >>> n
    ToReal(x)
    >>> n.sort()
    Real
    """
    if z3_debug():
        _z3_assert(a.is_int(), 'Z3 integer expression expected.')
    ctx = a.ctx
    return ArithRef(Z3_mk_int2real(ctx.ref(), a.as_ast()), ctx)