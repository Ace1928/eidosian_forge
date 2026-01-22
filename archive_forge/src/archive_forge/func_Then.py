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
def Then(*ts, **ks):
    """Return a tactic that applies the tactics in `*ts` in sequence. Shorthand for AndThen(*ts, **ks).

    >>> x, y = Ints('x y')
    >>> t = Then(Tactic('simplify'), Tactic('solve-eqs'))
    >>> t(And(x == 0, y > x + 1))
    [[Not(y <= 1)]]
    >>> t(And(x == 0, y > x + 1)).as_expr()
    Not(y <= 1)
    """
    return AndThen(*ts, **ks)