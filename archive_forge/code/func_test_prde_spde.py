from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_spde():
    D = [Poly(x, t), Poly(-x * t, t)]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    assert prde_spde(Poly(t, t), Poly(-1 / x, t), D, n, DE) == (Poly(t, t), Poly(0, t, domain='ZZ(x)'), [Poly(2 * x, t, domain='ZZ(x)'), Poly(-x, t, domain='ZZ(x)')], [Poly(-x ** 2, t, domain='ZZ(x)'), Poly(0, t, domain='ZZ(x)')], n - 1)