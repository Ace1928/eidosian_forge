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
def test_poly():
    assert poly(x) == Poly(x, x)
    assert poly(y) == Poly(y, y)
    assert poly(x + y) == Poly(x + y, x, y)
    assert poly(x + sin(x)) == Poly(x + sin(x), x, sin(x))
    assert poly(x + y, wrt=y) == Poly(x + y, y, x)
    assert poly(x + sin(x), wrt=sin(x)) == Poly(x + sin(x), sin(x), x)
    assert poly(x * y + 2 * x * z ** 2 + 17) == Poly(x * y + 2 * x * z ** 2 + 17, x, y, z)
    assert poly(2 * (y + z) ** 2 - 1) == Poly(2 * y ** 2 + 4 * y * z + 2 * z ** 2 - 1, y, z)
    assert poly(x * (y + z) ** 2 - 1) == Poly(x * y ** 2 + 2 * x * y * z + x * z ** 2 - 1, x, y, z)
    assert poly(2 * x * (y + z) ** 2 - 1) == Poly(2 * x * y ** 2 + 4 * x * y * z + 2 * x * z ** 2 - 1, x, y, z)
    assert poly(2 * (y + z) ** 2 - x - 1) == Poly(2 * y ** 2 + 4 * y * z + 2 * z ** 2 - x - 1, x, y, z)
    assert poly(x * (y + z) ** 2 - x - 1) == Poly(x * y ** 2 + 2 * x * y * z + x * z ** 2 - x - 1, x, y, z)
    assert poly(2 * x * (y + z) ** 2 - x - 1) == Poly(2 * x * y ** 2 + 4 * x * y * z + 2 * x * z ** 2 - x - 1, x, y, z)
    assert poly(x * y + (x + y) ** 2 + (x + z) ** 2) == Poly(2 * x * z + 3 * x * y + y ** 2 + z ** 2 + 2 * x ** 2, x, y, z)
    assert poly(x * y * (x + y) * (x + z) ** 2) == Poly(x ** 3 * y ** 2 + x * y ** 2 * z ** 2 + y * x ** 2 * z ** 2 + 2 * z * x ** 2 * y ** 2 + 2 * y * z * x ** 3 + y * x ** 4, x, y, z)
    assert poly(Poly(x + y + z, y, x, z)) == Poly(x + y + z, y, x, z)
    assert poly((x + y) ** 2, x) == Poly(x ** 2 + 2 * x * y + y ** 2, x, domain=ZZ[y])
    assert poly((x + y) ** 2, y) == Poly(x ** 2 + 2 * x * y + y ** 2, y, domain=ZZ[x])
    assert poly(1, x) == Poly(1, x)
    raises(GeneratorsNeeded, lambda: poly(1))
    assert poly(x + y, x, y) == Poly(x + y, x, y)
    assert poly(x + y, y, x) == Poly(x + y, y, x)