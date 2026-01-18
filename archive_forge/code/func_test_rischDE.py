from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    DE.decrement_level()
    assert rischDE(Poly(-2 * x, x), Poly(1, x), Poly(1 - 2 * x - 2 * x ** 2, x), Poly(1, x), DE) == (Poly(x + 1, x), Poly(1, x))