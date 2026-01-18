from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_cancel_liouvillian():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    p0 = Poly(0, t, field=True)
    p1 = Poly((x - 1) * t, t, domain='ZZ(x)')
    p2 = Poly(x - 1, t, domain='ZZ(x)')
    p3 = Poly(-x ** 2 + x, t, domain='ZZ(x)')
    h, A = prde_cancel_liouvillian(Poly(-1 / (x - 1), t), [Poly(-x + 1, t), Poly(1, t)], 1, DE)
    V = A.nullspace()
    assert h == [p0, p0, p1, p0, p0, p0, p0, p0, p0, p0, p2, p3, p0, p0, p0, p0]
    assert A.rank() == 16
    assert Matrix([h]) * V[0][:16, :] == Matrix([[Poly(0, t, domain='QQ(x)')]])
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t, t)]})
    assert prde_cancel_liouvillian(Poly(0, t, domain='QQ[x]'), [Poly(1, t, domain='QQ(x)')], 0, DE) == ([Poly(1, t, domain='QQ'), Poly(x, t, domain='ZZ(x)')], Matrix([[-1, 0, 1]], DE.t))