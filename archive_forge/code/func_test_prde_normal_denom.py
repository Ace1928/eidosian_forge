from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_normal_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    fa = Poly(1, t)
    fd = Poly(x, t)
    G = [(Poly(t, t), Poly(1 + t ** 2, t)), (Poly(1, t), Poly(x + x * t ** 2, t))]
    assert prde_normal_denom(fa, fd, G, DE) == (Poly(x, t, domain='ZZ(x)'), (Poly(1, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')), [(Poly(x * t, t, domain='ZZ(x)'), Poly(t ** 2 + 1, t, domain='ZZ(x)')), (Poly(1, t, domain='ZZ(x)'), Poly(t ** 2 + 1, t, domain='ZZ(x)'))], Poly(1, t, domain='ZZ(x)'))
    G = [(Poly(t, t), Poly(t ** 2 + 2 * t + 1, t)), (Poly(x * t, t), Poly(t ** 2 + 2 * t + 1, t)), (Poly(x * t ** 2, t), Poly(t ** 2 + 2 * t + 1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_normal_denom(Poly(x, t), Poly(1, t), G, DE) == (Poly(t + 1, t), (Poly((-1 + x) * t + x, t), Poly(1, t, domain='ZZ[x]')), [(Poly(t, t), Poly(1, t)), (Poly(x * t, t), Poly(1, t, domain='ZZ[x]')), (Poly(x * t ** 2, t), Poly(1, t, domain='ZZ[x]'))], Poly(t + 1, t))