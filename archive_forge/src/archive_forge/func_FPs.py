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
def FPs(names, fpsort, ctx=None):
    """Return an array of floating-point constants.

    >>> x, y, z = FPs('x y z', FPSort(8, 24))
    >>> x.sort()
    FPSort(8, 24)
    >>> x.sbits()
    24
    >>> x.ebits()
    8
    >>> fpMul(RNE(), fpAdd(RNE(), x, y), z)
    x + y * z
    """
    ctx = _get_ctx(ctx)
    if isinstance(names, str):
        names = names.split(' ')
    return [FP(name, fpsort, ctx) for name in names]