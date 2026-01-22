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
def arg_value(self, idx):
    """Return the value of argument `idx`.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0, 1) == 10, f(1, 2) == 20, f(1, 0) == 10)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> f_i = m[f]
        >>> f_i.num_entries()
        1
        >>> e = f_i.entry(0)
        >>> e
        [1, 2, 20]
        >>> e.num_args()
        2
        >>> e.arg_value(0)
        1
        >>> e.arg_value(1)
        2
        >>> try:
        ...   e.arg_value(2)
        ... except IndexError:
        ...   print("index error")
        index error
        """
    if idx >= self.num_args():
        raise IndexError
    return _to_expr_ref(Z3_func_entry_get_arg(self.ctx.ref(), self.entry, idx), self.ctx)