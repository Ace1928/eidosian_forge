from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_linear_constraints():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    G = [(Poly(2 * x ** 3 + 3 * x + 1, x), Poly(x ** 2 - 1, x)), (Poly(1, x), Poly(x - 1, x)), (Poly(1, x), Poly(x + 1, x))]
    assert prde_linear_constraints(Poly(1, x), Poly(0, x), G, DE) == ((Poly(2 * x, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(0, x, domain='QQ')), Matrix([[1, 1, -1], [5, 1, 1]], x))
    G = [(Poly(t, t), Poly(1, t)), (Poly(t ** 2, t), Poly(1, t)), (Poly(t ** 3, t), Poly(1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_linear_constraints(Poly(t + 1, t), Poly(t ** 2, t), G, DE) == ((Poly(t, t, domain='QQ'), Poly(t ** 2, t, domain='QQ'), Poly(t ** 3, t, domain='QQ')), Matrix(0, 3, [], t))
    G = [(Poly(2 * x, t), Poly(t, t)), (Poly(-x, t), Poly(t, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    assert prde_linear_constraints(Poly(1, t), Poly(0, t), G, DE) == ((Poly(0, t, domain='QQ[x]'), Poly(0, t, domain='QQ[x]')), Matrix([[2 * x, -x]], t))