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
def FPSort(ebits, sbits, ctx=None):
    """Return a Z3 floating-point sort of the given sizes. If `ctx=None`, then the global context is used.

    >>> Single = FPSort(8, 24)
    >>> Double = FPSort(11, 53)
    >>> Single
    FPSort(8, 24)
    >>> x = Const('x', Single)
    >>> eq(x, FP('x', FPSort(8, 24)))
    True
    """
    ctx = _get_ctx(ctx)
    return FPSortRef(Z3_mk_fpa_sort(ctx.ref(), ebits, sbits), ctx)