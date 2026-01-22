import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
def _try_heuristics(f):
    """Find roots using formulas and some tricks. """
    if f.is_ground:
        return []
    if f.is_monomial:
        return [S.Zero] * f.degree()
    if f.length() == 2:
        if f.degree() == 1:
            return list(map(cancel, roots_linear(f)))
        else:
            return roots_binomial(f)
    result = []
    for i in [-1, 1]:
        if not f.eval(i):
            f = f.quo(Poly(f.gen - i, f.gen))
            result.append(i)
            break
    n = f.degree()
    if n == 1:
        result += list(map(cancel, roots_linear(f)))
    elif n == 2:
        result += list(map(cancel, roots_quadratic(f)))
    elif f.is_cyclotomic:
        result += roots_cyclotomic(f)
    elif n == 3 and cubics:
        result += roots_cubic(f, trig=trig)
    elif n == 4 and quartics:
        result += roots_quartic(f)
    elif n == 5 and quintics:
        result += roots_quintic(f)
    return result