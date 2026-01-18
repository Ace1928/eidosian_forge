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
def test_real_roots():
    assert real_roots(x) == [0]
    assert real_roots(x, multiple=False) == [(0, 1)]
    assert real_roots(x ** 3) == [0, 0, 0]
    assert real_roots(x ** 3, multiple=False) == [(0, 3)]
    assert real_roots(x * (x ** 3 + x + 3)) == [rootof(x ** 3 + x + 3, 0), 0]
    assert real_roots(x * (x ** 3 + x + 3), multiple=False) == [(rootof(x ** 3 + x + 3, 0), 1), (0, 1)]
    assert real_roots(x ** 3 * (x ** 3 + x + 3)) == [rootof(x ** 3 + x + 3, 0), 0, 0, 0]
    assert real_roots(x ** 3 * (x ** 3 + x + 3), multiple=False) == [(rootof(x ** 3 + x + 3, 0), 1), (0, 3)]
    f = 2 * x ** 3 - 7 * x ** 2 + 4 * x + 4
    g = x ** 3 + x + 1
    assert Poly(f).real_roots() == [Rational(-1, 2), 2, 2]
    assert Poly(g).real_roots() == [rootof(g, 0)]