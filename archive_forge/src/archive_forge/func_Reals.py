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
def Reals(names, ctx=None):
    """Return a tuple of real constants.

    >>> x, y, z = Reals('x y z')
    >>> Sum(x, y, z)
    x + y + z
    >>> Sum(x, y, z).sort()
    Real
    """
    ctx = _get_ctx(ctx)
    if isinstance(names, str):
        names = names.split(' ')
    return [Real(name, ctx) for name in names]