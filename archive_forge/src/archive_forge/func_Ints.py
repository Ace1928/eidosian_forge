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
def Ints(names, ctx=None):
    """Return a tuple of Integer constants.

    >>> x, y, z = Ints('x y z')
    >>> Sum(x, y, z)
    x + y + z
    """
    ctx = _get_ctx(ctx)
    if isinstance(names, str):
        names = names.split(' ')
    return [Int(name, ctx) for name in names]