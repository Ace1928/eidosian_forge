from sympy.polys.domains import QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, atan, atanh,
from sympy.core.numbers import Rational
from sympy.core import expand, S
def test_puiseux():
    R, x, y = ring('x, y', QQ)
    p = x ** QQ(2, 5) + x ** QQ(2, 3) + x
    r = rs_series_inversion(p, x, 1)
    r1 = -x ** QQ(14, 15) + x ** QQ(4, 5) - 3 * x ** QQ(11, 15) + x ** QQ(2, 3) + 2 * x ** QQ(7, 15) - x ** QQ(2, 5) - x ** QQ(1, 5) + x ** QQ(2, 15) - x ** QQ(-2, 15) + x ** QQ(-2, 5)
    assert r == r1
    r = rs_nth_root(1 + p, 3, x, 1)
    assert r == -x ** QQ(4, 5) / 9 + x ** QQ(2, 3) / 3 + x ** QQ(2, 5) / 3 + 1
    r = rs_log(1 + p, x, 1)
    assert r == -x ** QQ(4, 5) / 2 + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_LambertW(p, x, 1)
    assert r == -x ** QQ(4, 5) + x ** QQ(2, 3) + x ** QQ(2, 5)
    p1 = x + x ** QQ(1, 5) * y
    r = rs_exp(p1, x, 1)
    assert r == x ** QQ(4, 5) * y ** 4 / 24 + x ** QQ(3, 5) * y ** 3 / 6 + x ** QQ(2, 5) * y ** 2 / 2 + x ** QQ(1, 5) * y + 1
    r = rs_atan(p, x, 2)
    assert r == -x ** QQ(9, 5) - x ** QQ(26, 15) - x ** QQ(22, 15) - x ** QQ(6, 5) / 3 + x + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_atan(p1, x, 2)
    assert r == x ** QQ(9, 5) * y ** 9 / 9 + x ** QQ(9, 5) * y ** 4 - x ** QQ(7, 5) * y ** 7 / 7 - x ** QQ(7, 5) * y ** 2 + x * y ** 5 / 5 + x - x ** QQ(3, 5) * y ** 3 / 3 + x ** QQ(1, 5) * y
    r = rs_asin(p, x, 2)
    assert r == x ** QQ(9, 5) / 2 + x ** QQ(26, 15) / 2 + x ** QQ(22, 15) / 2 + x ** QQ(6, 5) / 6 + x + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_cot(p, x, 1)
    assert r == -x ** QQ(14, 15) + x ** QQ(4, 5) - 3 * x ** QQ(11, 15) + 2 * x ** QQ(2, 3) / 3 + 2 * x ** QQ(7, 15) - 4 * x ** QQ(2, 5) / 3 - x ** QQ(1, 5) + x ** QQ(2, 15) - x ** QQ(-2, 15) + x ** QQ(-2, 5)
    r = rs_cos_sin(p, x, 2)
    assert r[0] == x ** QQ(28, 15) / 6 - x ** QQ(5, 3) + x ** QQ(8, 5) / 24 - x ** QQ(7, 5) - x ** QQ(4, 3) / 2 - x ** QQ(16, 15) - x ** QQ(4, 5) / 2 + 1
    assert r[1] == -x ** QQ(9, 5) / 2 - x ** QQ(26, 15) / 2 - x ** QQ(22, 15) / 2 - x ** QQ(6, 5) / 6 + x + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_atanh(p, x, 2)
    assert r == x ** QQ(9, 5) + x ** QQ(26, 15) + x ** QQ(22, 15) + x ** QQ(6, 5) / 3 + x + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_sinh(p, x, 2)
    assert r == x ** QQ(9, 5) / 2 + x ** QQ(26, 15) / 2 + x ** QQ(22, 15) / 2 + x ** QQ(6, 5) / 6 + x + x ** QQ(2, 3) + x ** QQ(2, 5)
    r = rs_cosh(p, x, 2)
    assert r == x ** QQ(28, 15) / 6 + x ** QQ(5, 3) + x ** QQ(8, 5) / 24 + x ** QQ(7, 5) + x ** QQ(4, 3) / 2 + x ** QQ(16, 15) + x ** QQ(4, 5) / 2 + 1
    r = rs_tanh(p, x, 2)
    assert r == -x ** QQ(9, 5) - x ** QQ(26, 15) - x ** QQ(22, 15) - x ** QQ(6, 5) / 3 + x + x ** QQ(2, 3) + x ** QQ(2, 5)