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
def test_terms_gcd():
    assert terms_gcd(1) == 1
    assert terms_gcd(1, x) == 1
    assert terms_gcd(x - 1) == x - 1
    assert terms_gcd(-x - 1) == -x - 1
    assert terms_gcd(2 * x + 3) == 2 * x + 3
    assert terms_gcd(6 * x + 4) == Mul(2, 3 * x + 2, evaluate=False)
    assert terms_gcd(x ** 3 * y + x * y ** 3) == x * y * (x ** 2 + y ** 2)
    assert terms_gcd(2 * x ** 3 * y + 2 * x * y ** 3) == 2 * x * y * (x ** 2 + y ** 2)
    assert terms_gcd(x ** 3 * y / 2 + x * y ** 3 / 2) == x * y / 2 * (x ** 2 + y ** 2)
    assert terms_gcd(x ** 3 * y + 2 * x * y ** 3) == x * y * (x ** 2 + 2 * y ** 2)
    assert terms_gcd(2 * x ** 3 * y + 4 * x * y ** 3) == 2 * x * y * (x ** 2 + 2 * y ** 2)
    assert terms_gcd(2 * x ** 3 * y / 3 + 4 * x * y ** 3 / 5) == x * y * Rational(2, 15) * (5 * x ** 2 + 6 * y ** 2)
    assert terms_gcd(2.0 * x ** 3 * y + 4.1 * x * y ** 3) == x * y * (2.0 * x ** 2 + 4.1 * y ** 2)
    assert _aresame(terms_gcd(2.0 * x + 3), 2.0 * x + 3)
    assert terms_gcd((3 + 3 * x) * (x + x * y), expand=False) == (3 * x + 3) * (x * y + x)
    assert terms_gcd((3 + 3 * x) * (x + x * sin(3 + 3 * y)), expand=False, deep=True) == 3 * x * (x + 1) * (sin(Mul(3, y + 1, evaluate=False)) + 1)
    assert terms_gcd(sin(x + x * y), deep=True) == sin(x * (y + 1))
    eq = Eq(2 * x, 2 * y + 2 * z * y)
    assert terms_gcd(eq) == Eq(2 * x, 2 * y * (z + 1))
    assert terms_gcd(eq, deep=True) == Eq(2 * x, 2 * y * (z + 1))
    raises(TypeError, lambda: terms_gcd(x < 2))