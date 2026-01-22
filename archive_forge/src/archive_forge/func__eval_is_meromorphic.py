from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def _eval_is_meromorphic(self, x, a):
    if not a.is_real:
        return None
    for e, c in self.args:
        cond = c.subs(x, a)
        if cond.is_Relational:
            return None
        if a in c.as_set().boundary:
            return None
        if cond:
            return e._eval_is_meromorphic(x, a)