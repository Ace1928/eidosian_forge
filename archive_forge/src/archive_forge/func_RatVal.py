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
def RatVal(a, b, ctx=None):
    """Return a Z3 rational a/b.

    If `ctx=None`, then the global context is used.

    >>> RatVal(3,5)
    3/5
    >>> RatVal(3,5).sort()
    Real
    """
    if z3_debug():
        _z3_assert(_is_int(a) or isinstance(a, str), 'First argument cannot be converted into an integer')
        _z3_assert(_is_int(b) or isinstance(b, str), 'Second argument cannot be converted into an integer')
    return simplify(RealVal(a, ctx) / RealVal(b, ctx))