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
def _piecewise_collapse_arguments(_args):
    newargs = []
    current_cond = set()
    for expr, cond in _args:
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        if isinstance(expr, Piecewise):
            unmatching = []
            for i, (e, c) in enumerate(expr.args):
                if c in current_cond:
                    continue
                if c == cond:
                    if c != True:
                        if unmatching:
                            expr = Piecewise(*unmatching + [(e, c)])
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))
        got = False
        for i in [cond] + (list(cond.args) if isinstance(cond, And) else []):
            if i in current_cond:
                got = True
                break
        if got:
            continue
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if isinstance(c, Relational):
                    if c.negated.canonical in current_cond:
                        continue
                    if isinstance(c, (Lt, Gt)) and c.weak in current_cond:
                        cond = False
                        break
                nonredundant.append(c)
            else:
                cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current_cond:
                cond = S.true
        current_cond.add(cond)
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs