from sympy.core import cacheit, Dummy, Ne, Integer, Rational, S, Wild
from sympy.functions import binomial, sin, cos, Piecewise, Abs
from .integrals import integrate
def _sin_pow_integrate(n, x):
    if n > 0:
        if n == 1:
            return -cos(x)
        return Rational(-1, n) * cos(x) * sin(x) ** (n - 1) + Rational(n - 1, n) * _sin_pow_integrate(n - 2, x)
    if n < 0:
        if n == -1:
            return trigintegrate(1 / sin(x), x)
        return Rational(1, n + 1) * cos(x) * sin(x) ** (n + 1) + Rational(n + 2, n + 1) * _sin_pow_integrate(n + 2, x)
    else:
        return x