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
def ToInt(a):
    """ Return the Z3 expression ToInt(a).

    >>> x = Real('x')
    >>> x.sort()
    Real
    >>> n = ToInt(x)
    >>> n
    ToInt(x)
    >>> n.sort()
    Int
    """
    if z3_debug():
        _z3_assert(a.is_real(), 'Z3 real expression expected.')
    ctx = a.ctx
    return ArithRef(Z3_mk_real2int(ctx.ref(), a.as_ast()), ctx)