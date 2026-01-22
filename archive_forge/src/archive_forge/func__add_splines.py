from sympy.core import S, sympify
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from functools import lru_cache
def _add_splines(c, b1, d, b2, x):
    """Construct c*b1 + d*b2."""
    if S.Zero in (b1, c):
        rv = piecewise_fold(d * b2)
    elif S.Zero in (b2, d):
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)
        p2args = list(p2.args[:-1])
        for arg in p1.args[:-1]:
            expr = arg.expr
            cond = arg.cond
            lower = _ivl(cond, x)[0]
            for i, arg2 in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond
                lower_2, upper_2 = _ivl(cond2, x)
                if cond2 == cond:
                    expr += expr2
                    del p2args[i]
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    new_args.append(arg2)
                    del p2args[i]
                    break
            new_args.append((expr, cond))
        new_args.extend(p2args)
        new_args.append((0, True))
        rv = Piecewise(*new_args, evaluate=False)
    return rv.expand()