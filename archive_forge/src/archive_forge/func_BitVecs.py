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
def BitVecs(names, bv, ctx=None):
    """Return a tuple of bit-vector constants of size bv.

    >>> x, y, z = BitVecs('x y z', 16)
    >>> x.size()
    16
    >>> x.sort()
    BitVec(16)
    >>> Sum(x, y, z)
    0 + x + y + z
    >>> Product(x, y, z)
    1*x*y*z
    >>> simplify(Product(x, y, z))
    x*y*z
    """
    ctx = _get_ctx(ctx)
    if isinstance(names, str):
        names = names.split(' ')
    return [BitVec(name, bv, ctx) for name in names]