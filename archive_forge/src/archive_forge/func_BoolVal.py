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
def BoolVal(val, ctx=None):
    """Return the Boolean value `True` or `False`. If `ctx=None`, then the global context is used.

    >>> BoolVal(True)
    True
    >>> is_true(BoolVal(True))
    True
    >>> is_true(True)
    False
    >>> is_false(BoolVal(False))
    True
    """
    ctx = _get_ctx(ctx)
    if val:
        return BoolRef(Z3_mk_true(ctx.ref()), ctx)
    else:
        return BoolRef(Z3_mk_false(ctx.ref()), ctx)