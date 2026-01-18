from sympy.core import Symbol
from sympy.core.evalf import N
from sympy.core.numbers import I, Rational
from sympy.functions import sqrt
from sympy.polys.polytools import Poly
from sympy.utilities import public
def uv(self, theta, d):
    c = self.c
    u = self.q * Rational(-25, 2)
    v = Poly(c, x).eval(theta) / (2 * d * self.F)
    return (N(u), N(v))