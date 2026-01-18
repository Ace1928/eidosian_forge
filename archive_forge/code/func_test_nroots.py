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
def test_nroots():
    assert Poly(0, x).nroots() == []
    assert Poly(1, x).nroots() == []
    assert Poly(x ** 2 - 1, x).nroots() == [-1.0, 1.0]
    assert Poly(x ** 2 + 1, x).nroots() == [-1.0 * I, 1.0 * I]
    roots = Poly(x ** 2 - 1, x).nroots()
    assert roots == [-1.0, 1.0]
    roots = Poly(x ** 2 + 1, x).nroots()
    assert roots == [-1.0 * I, 1.0 * I]
    roots = Poly(x ** 2 / 3 - Rational(1, 3), x).nroots()
    assert roots == [-1.0, 1.0]
    roots = Poly(x ** 2 / 3 + Rational(1, 3), x).nroots()
    assert roots == [-1.0 * I, 1.0 * I]
    assert Poly(x ** 2 + 2 * I, x).nroots() == [-1.0 + 1.0 * I, 1.0 - 1.0 * I]
    assert Poly(x ** 2 + 2 * I, x, extension=I).nroots() == [-1.0 + 1.0 * I, 1.0 - 1.0 * I]
    assert Poly(0.2 * x + 0.1).nroots() == [-0.5]
    roots = nroots(x ** 5 + x + 1, n=5)
    eps = Float('1e-5')
    assert re(roots[0]).epsilon_eq(-0.75487, eps) is S.true
    assert im(roots[0]) == 0.0
    assert re(roots[1]) == Float(-0.5, 5)
    assert im(roots[1]).epsilon_eq(-0.86602, eps) is S.true
    assert re(roots[2]) == Float(-0.5, 5)
    assert im(roots[2]).epsilon_eq(+0.86602, eps) is S.true
    assert re(roots[3]).epsilon_eq(+0.87743, eps) is S.true
    assert im(roots[3]).epsilon_eq(-0.74486, eps) is S.true
    assert re(roots[4]).epsilon_eq(+0.87743, eps) is S.true
    assert im(roots[4]).epsilon_eq(+0.74486, eps) is S.true
    eps = Float('1e-6')
    assert re(roots[0]).epsilon_eq(-0.75487, eps) is S.false
    assert im(roots[0]) == 0.0
    assert re(roots[1]) == Float(-0.5, 5)
    assert im(roots[1]).epsilon_eq(-0.86602, eps) is S.false
    assert re(roots[2]) == Float(-0.5, 5)
    assert im(roots[2]).epsilon_eq(+0.86602, eps) is S.false
    assert re(roots[3]).epsilon_eq(+0.87743, eps) is S.false
    assert im(roots[3]).epsilon_eq(-0.74486, eps) is S.false
    assert re(roots[4]).epsilon_eq(+0.87743, eps) is S.false
    assert im(roots[4]).epsilon_eq(+0.74486, eps) is S.false
    raises(DomainError, lambda: Poly(x + y, x).nroots())
    raises(MultivariatePolynomialError, lambda: Poly(x + y).nroots())
    assert nroots(x ** 2 - 1) == [-1.0, 1.0]
    roots = nroots(x ** 2 - 1)
    assert roots == [-1.0, 1.0]
    assert nroots(x + I) == [-1.0 * I]
    assert nroots(x + 2 * I) == [-2.0 * I]
    raises(PolynomialError, lambda: nroots(0))
    f = Poly(x ** 4 - 1)
    assert f.nroots(2) == [w.n(2) for w in f.all_roots()]
    assert str(Poly(x ** 16 + 32 * x ** 14 + 508 * x ** 12 + 5440 * x ** 10 + 39510 * x ** 8 + 204320 * x ** 6 + 755548 * x ** 4 + 1434496 * x ** 2 + 877969).nroots(2)) == '[-1.7 - 1.9*I, -1.7 + 1.9*I, -1.7 - 2.5*I, -1.7 + 2.5*I, -1.0*I, 1.0*I, -1.7*I, 1.7*I, -2.8*I, 2.8*I, -3.4*I, 3.4*I, 1.7 - 1.9*I, 1.7 + 1.9*I, 1.7 - 2.5*I, 1.7 + 2.5*I]'
    assert str(Poly(1e-15 * x ** 2 - 1).nroots()) == '[-31622776.6016838, 31622776.6016838]'