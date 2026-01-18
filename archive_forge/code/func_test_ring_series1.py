from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_ring_series1():
    R, x = ring('x', QQ)
    p = x ** 4 + 2 * x ** 3 + 3 * x + 4
    assert _invert_monoms(p) == 4 * x ** 4 + 3 * x ** 3 + 2 * x + 1
    assert rs_hadamard_exp(p) == x ** 4 / 24 + x ** 3 / 3 + 3 * x + 4
    R, x = ring('x', QQ)
    p = x ** 4 + 2 * x ** 3 + 3 * x + 4
    assert rs_integrate(p, x) == x ** 5 / 5 + x ** 4 / 2 + 3 * x ** 2 / 2 + 4 * x
    R, x, y = ring('x, y', QQ)
    p = x ** 2 * y ** 2 + x + 1
    assert rs_integrate(p, x) == x ** 3 * y ** 2 / 3 + x ** 2 / 2 + x
    assert rs_integrate(p, y) == x ** 2 * y ** 3 / 3 + x * y + y