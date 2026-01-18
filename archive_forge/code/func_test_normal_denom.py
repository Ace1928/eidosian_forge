from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_normal_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    raises(NonElementaryIntegralException, lambda: normal_denom(Poly(1, x), Poly(1, x), Poly(1, x), Poly(x, x), DE))
    fa, fd = (Poly(t ** 2 + 1, t), Poly(1, t))
    ga, gd = (Poly(1, t), Poly(t ** 2, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert normal_denom(fa, fd, ga, gd, DE) == (Poly(t, t), (Poly(t ** 3 - t ** 2 + t - 1, t), Poly(1, t)), (Poly(1, t), Poly(1, t)), Poly(t, t))