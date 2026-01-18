from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_parametric_log_deriv():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    assert parametric_log_deriv_heu(Poly(5 * t ** 2 + t - 6, t), Poly(2 * x * t ** 2, t), Poly(-1, t), Poly(x * t ** 2, t), DE) == (2, 6, t * x ** 5)