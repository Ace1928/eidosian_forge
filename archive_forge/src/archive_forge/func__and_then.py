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
def _and_then(t1, t2, ctx=None):
    t1 = _to_tactic(t1, ctx)
    t2 = _to_tactic(t2, ctx)
    if z3_debug():
        _z3_assert(t1.ctx == t2.ctx, 'Context mismatch')
    return Tactic(Z3_tactic_and_then(t1.ctx.ref(), t1.tactic, t2.tactic), t1.ctx)