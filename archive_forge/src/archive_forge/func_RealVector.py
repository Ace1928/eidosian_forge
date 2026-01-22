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
def RealVector(prefix, sz, ctx=None):
    """Return a list of real constants of size `sz`.

    >>> X = RealVector('x', 3)
    >>> X
    [x__0, x__1, x__2]
    >>> Sum(X)
    x__0 + x__1 + x__2
    >>> Sum(X).sort()
    Real
    """
    ctx = _get_ctx(ctx)
    return [Real('%s__%s' % (prefix, i), ctx) for i in range(sz)]