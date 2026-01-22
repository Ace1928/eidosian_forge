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
def AndThen(*ts, **ks):
    """Return a tactic that applies the tactics in `*ts` in sequence.

    >>> x, y = Ints('x y')
    >>> t = AndThen(Tactic('simplify'), Tactic('solve-eqs'))
    >>> t(And(x == 0, y > x + 1))
    [[Not(y <= 1)]]
    >>> t(And(x == 0, y > x + 1)).as_expr()
    Not(y <= 1)
    """
    if z3_debug():
        _z3_assert(len(ts) >= 2, 'At least two arguments expected')
    ctx = ks.get('ctx', None)
    num = len(ts)
    r = ts[0]
    for i in range(num - 1):
        r = _and_then(r, ts[i + 1], ctx)
    return r