from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_order_at():
    a = Poly(t ** 4, t)
    b = Poly((t ** 2 + 1) ** 3 * t, t)
    c = Poly((t ** 2 + 1) ** 6 * t, t)
    d = Poly((t ** 2 + 1) ** 10 * t ** 10, t)
    e = Poly((t ** 2 + 1) ** 100 * t ** 37, t)
    p1 = Poly(t, t)
    p2 = Poly(1 + t ** 2, t)
    assert order_at(a, p1, t) == 4
    assert order_at(b, p1, t) == 1
    assert order_at(c, p1, t) == 1
    assert order_at(d, p1, t) == 10
    assert order_at(e, p1, t) == 37
    assert order_at(a, p2, t) == 0
    assert order_at(b, p2, t) == 3
    assert order_at(c, p2, t) == 6
    assert order_at(d, p1, t) == 10
    assert order_at(e, p2, t) == 100
    assert order_at(Poly(0, t), Poly(t, t), t) is oo
    assert order_at_oo(Poly(t ** 2 - 1, t), Poly(t + 1), t) == order_at_oo(Poly(t - 1, t), Poly(1, t), t) == -1
    assert order_at_oo(Poly(0, t), Poly(1, t), t) is oo