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
def var_sort(self, idx):
    """Return the sort of a bound variable.

        >>> f = Function('f', IntSort(), RealSort(), IntSort())
        >>> x = Int('x')
        >>> y = Real('y')
        >>> q = ForAll([x, y], f(x, y) >= x)
        >>> q.var_sort(0)
        Int
        >>> q.var_sort(1)
        Real
        """
    if z3_debug():
        _z3_assert(idx < self.num_vars(), 'Invalid variable idx')
    return _to_sort_ref(Z3_get_quantifier_bound_sort(self.ctx_ref(), self.ast, idx), self.ctx)