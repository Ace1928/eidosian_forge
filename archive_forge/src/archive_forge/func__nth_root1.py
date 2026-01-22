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
def _nth_root1(p, n, x, prec):
    """
    Univariate series expansion of the nth root of ``p``.

    The Newton method is used.
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux2(_nth_root1, p, n, x, prec)
    R = p.ring
    zm = R.zero_monom
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    n = as_int(n)
    assert p[zm] == 1
    p1 = R(1)
    if p == 1:
        return p
    if n == 0:
        return R(1)
    if n == 1:
        return p
    if n < 0:
        n = -n
        sign = 1
    else:
        sign = 0
    for precx in _giant_steps(prec):
        tmp = rs_pow(p1, n + 1, x, precx)
        tmp = rs_mul(tmp, p, x, precx)
        p1 += p1 / n - tmp / n
    if sign:
        return p1
    else:
        return _series_inversion1(p1, x, prec)