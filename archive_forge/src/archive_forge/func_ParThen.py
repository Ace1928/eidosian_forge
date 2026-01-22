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
def ParThen(t1, t2, ctx=None):
    """Return a tactic that applies t1 and then t2 to every subgoal produced by t1.
    The subgoals are processed in parallel.

    >>> x, y = Ints('x y')
    >>> t = ParThen(Tactic('split-clause'), Tactic('propagate-values'))
    >>> t(And(Or(x == 1, x == 2), y == x + 1))
    [[x == 1, y == 2], [x == 2, y == 3]]
    """
    t1 = _to_tactic(t1, ctx)
    t2 = _to_tactic(t2, ctx)
    if z3_debug():
        _z3_assert(t1.ctx == t2.ctx, 'Context mismatch')
    return Tactic(Z3_tactic_par_and_then(t1.ctx.ref(), t1.tactic, t2.tactic), t1.ctx)