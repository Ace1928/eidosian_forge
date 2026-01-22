from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
    opt = build_options(args)
    gens_order, wrt = ({}, None)
    if opt is not None:
        gens_order, wrt = ({}, opt.wrt)
        for i, gen in enumerate(opt.sort):
            gens_order[gen] = i + 1

    def order_key(gen):
        gen = str(gen)
        if wrt is not None:
            try:
                return (-len(wrt) + wrt.index(gen), gen, 0)
            except ValueError:
                pass
        name, index = _re_gen.match(gen).groups()
        if index:
            index = int(index)
        else:
            index = 0
        try:
            return (gens_order[name], name, index)
        except KeyError:
            pass
        try:
            return (_gens_order[name], name, index)
        except KeyError:
            pass
        return (_max_order, name, index)
    try:
        gens = sorted(gens, key=order_key)
    except TypeError:
        pass
    return tuple(gens)