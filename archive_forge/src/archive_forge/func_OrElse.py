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
def OrElse(*ts, **ks):
    """Return a tactic that applies the tactics in `*ts` until one of them succeeds (it doesn't fail).

    >>> x = Int('x')
    >>> t = OrElse(Tactic('split-clause'), Tactic('skip'))
    >>> # Tactic split-clause fails if there is no clause in the given goal.
    >>> t(x == 0)
    [[x == 0]]
    >>> t(Or(x == 0, x == 1))
    [[x == 0], [x == 1]]
    """
    if z3_debug():
        _z3_assert(len(ts) >= 2, 'At least two arguments expected')
    ctx = ks.get('ctx', None)
    num = len(ts)
    r = ts[0]
    for i in range(num - 1):
        r = _or_else(r, ts[i + 1], ctx)
    return r