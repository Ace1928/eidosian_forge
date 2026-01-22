from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def _get_constant_term(p, x):
    """Return constant term in p with respect to x

    Note that it is not simply `p[R.zero_monom]` as there might be multiple
    generators in the ring R. We want the `x`-free term which can contain other
    generators.
    """
    R = p.ring
    i = R.gens.index(x)
    zm = R.zero_monom
    a = [0] * R.ngens
    a[i] = 1
    miv = tuple(a)
    c = 0
    for expv in p:
        if monomial_min(expv, miv) == zm:
            c += R({expv: p[expv]})
    return c