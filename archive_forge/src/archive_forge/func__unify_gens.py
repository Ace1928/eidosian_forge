from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _unify_gens(f_gens, g_gens):
    """Unify generators in a reasonably intelligent way. """
    f_gens = list(f_gens)
    g_gens = list(g_gens)
    if f_gens == g_gens:
        return tuple(f_gens)
    gens, common, k = ([], [], 0)
    for gen in f_gens:
        if gen in g_gens:
            common.append(gen)
    for i, gen in enumerate(g_gens):
        if gen in common:
            g_gens[i], k = (common[k], k + 1)
    for gen in common:
        i = f_gens.index(gen)
        gens.extend(f_gens[:i])
        f_gens = f_gens[i + 1:]
        i = g_gens.index(gen)
        gens.extend(g_gens[:i])
        g_gens = g_gens[i + 1:]
        gens.append(gen)
    gens.extend(f_gens)
    gens.extend(g_gens)
    return tuple(gens)