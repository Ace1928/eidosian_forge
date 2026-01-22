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
def add_soft(self, arg, weight='1', id=None):
    """Add soft constraint with optional weight and optional identifier.
           If no weight is supplied, then the penalty for violating the soft constraint
           is 1.
           Soft constraints are grouped by identifiers. Soft constraints that are
           added without identifiers are grouped by default.
        """
    if _is_int(weight):
        weight = '%d' % weight
    elif isinstance(weight, float):
        weight = '%f' % weight
    if not isinstance(weight, str):
        raise Z3Exception('weight should be a string or an integer')
    if id is None:
        id = ''
    id = to_symbol(id, self.ctx)

    def asoft(a):
        v = Z3_optimize_assert_soft(self.ctx.ref(), self.optimize, a.as_ast(), weight, id)
        return OptimizeObjective(self, v, False)
    if sys.version_info.major >= 3 and isinstance(arg, Iterable):
        return [asoft(a) for a in arg]
    return asoft(arg)