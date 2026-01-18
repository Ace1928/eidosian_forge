import pickle
from sympy.polys.polytools import (
from sympy.polys.polyerrors import (
from sympy.polys.polyclasses import DMP
from sympy.polys.fields import field
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.orderings import lex, grlex, grevlex
from sympy.combinatorics.galois import S4TransitiveSubgroups
from sympy.core.add import Add
from sympy.core.basic import _aresame
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, diff, expand)
from sympy.core.mul import _keep_coeff, Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.rootoftools import rootof
from sympy.simplify.simplify import signsimp
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.testing.pytest import raises, warns_deprecated_sympy, warns
from sympy.abc import a, b, c, d, p, q, t, w, x, y, z
def test_sturm():
    f, F = (x, Poly(x, domain='QQ'))
    g, G = (1, Poly(1, x, domain='QQ'))
    assert F.sturm() == [F, G]
    assert sturm(f) == [f, g]
    assert sturm(f, x) == [f, g]
    assert sturm(f, (x,)) == [f, g]
    assert sturm(F) == [F, G]
    assert sturm(f, polys=True) == [F, G]
    assert sturm(F, polys=False) == [f, g]
    raises(ComputationFailed, lambda: sturm(4))
    raises(DomainError, lambda: sturm(f, auto=False))
    f = Poly(S(1024) / (15625 * pi ** 8) * x ** 5 - S(4096) / (625 * pi ** 8) * x ** 4 + S(32) / (15625 * pi ** 4) * x ** 3 - S(128) / (625 * pi ** 4) * x ** 2 + Rational(1, 62500) * x - Rational(1, 625), x, domain='ZZ(pi)')
    assert sturm(f) == [Poly(x ** 3 - 100 * x ** 2 + pi ** 4 / 64 * x - 25 * pi ** 4 / 16, x, domain='ZZ(pi)'), Poly(3 * x ** 2 - 200 * x + pi ** 4 / 64, x, domain='ZZ(pi)'), Poly((Rational(20000, 9) - pi ** 4 / 96) * x + 25 * pi ** 4 / 18, x, domain='ZZ(pi)'), Poly((-3686400000000 * pi ** 4 - 11520000 * pi ** 8 - 9 * pi ** 12) / (26214400000000 - 245760000 * pi ** 4 + 576 * pi ** 8), x, domain='ZZ(pi)')]