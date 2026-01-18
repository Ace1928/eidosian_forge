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
def test_refine_root():
    f = Poly(x ** 2 - 2)
    assert f.refine_root(1, 2, steps=0) == (1, 2)
    assert f.refine_root(-2, -1, steps=0) == (-2, -1)
    assert f.refine_root(1, 2, steps=None) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=None) == (Rational(-3, 2), -1)
    assert f.refine_root(1, 2, steps=1) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1) == (Rational(-3, 2), -1)
    assert f.refine_root(1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)
    assert f.refine_root(1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert f.refine_root(1, 2, eps=0.01) == (Rational(24, 17), Rational(17, 12))
    raises(PolynomialError, lambda: (f ** 2).refine_root(1, 2, check_sqf=True))
    raises(RefinementFailed, lambda: (f ** 2).refine_root(1, 2))
    raises(RefinementFailed, lambda: (f ** 2).refine_root(2, 3))
    f = x ** 2 - 2
    assert refine_root(f, 1, 2, steps=1) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1) == (Rational(-3, 2), -1)
    assert refine_root(f, 1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)
    assert refine_root(f, 1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert refine_root(f, 1, 2, eps=0.01) == (Rational(24, 17), Rational(17, 12))
    raises(PolynomialError, lambda: refine_root(1, 7, 8, eps=Rational(1, 100)))
    raises(ValueError, lambda: Poly(f).refine_root(1, 2, eps=10 ** (-100000)))
    raises(ValueError, lambda: refine_root(f, 1, 2, eps=10 ** (-100000)))