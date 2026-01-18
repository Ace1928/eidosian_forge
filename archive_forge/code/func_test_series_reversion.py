from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_series_reversion():
    R, x, y = ring('x, y', QQ)
    p = rs_tan(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == rs_atan(y, y, 8)
    p = rs_sin(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == 5 * y ** 7 / 112 + 3 * y ** 5 / 40 + y ** 3 / 6 + y