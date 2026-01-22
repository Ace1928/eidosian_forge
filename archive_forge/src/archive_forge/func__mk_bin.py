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
def _mk_bin(f, a, b):
    args = (Ast * 2)()
    if z3_debug():
        _z3_assert(a.ctx == b.ctx, 'Context mismatch')
    args[0] = a.as_ast()
    args[1] = b.as_ast()
    return f(a.ctx.ref(), 2, args)