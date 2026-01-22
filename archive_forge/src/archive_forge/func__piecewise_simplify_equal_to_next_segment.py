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
def _piecewise_simplify_equal_to_next_segment(args):
    """
    See if expressions valid for an Equal expression happens to evaluate
    to the same function as in the next piecewise segment, see:
    https://github.com/sympy/sympy/issues/8458
    """
    prevexpr = None
    for i, (expr, cond) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            if isinstance(cond, And):
                eqs, other = sift(cond.args, lambda i: isinstance(i, Eq), binary=True)
            elif isinstance(cond, Eq):
                eqs, other = ([cond], [])
            else:
                eqs = other = []
            _prevexpr = prevexpr
            _expr = expr
            if eqs and (not other):
                eqs = list(ordered(eqs))
                for e in eqs:
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _prevexpr.subs(*e.args)
                        _expr = _expr.subs(*e.args)
            if _prevexpr == _expr:
                args[i] = args[i].func(args[i + 1][0], cond)
            else:
                prevexpr = expr
        else:
            prevexpr = expr
    return args