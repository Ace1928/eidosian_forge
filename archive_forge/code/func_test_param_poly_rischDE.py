from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_param_poly_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    a = Poly(x ** 2 - x, x, field=True)
    b = Poly(1, x, field=True)
    q = [Poly(x, x, field=True), Poly(x ** 2, x, field=True)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)
    assert A.nullspace() == [Matrix([0, 1, 1, 1], DE.t)]
    assert h[0] + h[1] == Poly(x, x, domain='QQ')
    a = Poly(x ** 2 - x, x, field=True)
    b = Poly(x ** 2 - 5 * x + 3, x, field=True)
    q = [Poly(1, x, field=True), Poly(x, x, field=True), Poly(x ** 2, x, field=True)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)
    assert A.nullspace() == [Matrix([3, -5, 1, -5, 1, 1], DE.t)]
    p = -Poly(5, DE.t) * h[0] + h[1] + h[2]
    assert a * derivation(p, DE) + b * p == Poly(x ** 2 - 5 * x + 3, x, domain='QQ')