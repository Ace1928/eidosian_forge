from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import (Derivative, Function, Lambda, Subs)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.core.sympify import SympifyError
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, cot, sin, tan)
from sympy.matrices.dense import (Matrix, zeros)
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.polys.polytools import factor
from sympy.polys.rootoftools import RootOf
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import nsimplify
from sympy.core.basic import _aresame
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import a, x, y, z, t
def test_subs_noncommutative():
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    alpha = symbols('alpha', commutative=True)
    someint = symbols('someint', commutative=True, integer=True)
    assert (x * y).subs(x * y, L) == L
    assert (w * y * x).subs(x * y, L) == w * y * x
    assert (w * x * y * z).subs(x * y, L) == w * L * z
    assert (x * y * x * y).subs(x * y, L) == L ** 2
    assert (x * x * y).subs(x * y, L) == x * L
    assert (x * x * y * y).subs(x * y, L) == x * L * y
    assert (w * x * y).subs(x * y * z, L) == w * x * y
    assert (x * y ** z).subs(x, L) == L * y ** z
    assert (x * y ** z).subs(y, L) == x * L ** z
    assert (x * y ** z).subs(z, L) == x * y ** L
    assert (w * x * y * z * x * y).subs(x * y * z, L) == w * L * x * y
    assert (w * x * y * y * w * x * x * y * x * y * y * x * y).subs(x * y, L) == w * L * y * w * x * L ** 2 * y * L
    assert (x * x * x).subs(x * x, L) == L * x
    assert (x * x * x * y * x * x * x * x).subs(x * x, L) == L * x * y * L ** 2
    for p in range(1, 5):
        for k in range(10):
            assert (y * x ** k).subs(x ** p, L) == y * L ** (k // p) * x ** (k % p)
    assert (x ** Rational(3, 2)).subs(x ** S.Half, L) == x ** Rational(3, 2)
    assert (x ** S.Half).subs(x ** S.Half, L) == L
    assert (x ** Rational(-1, 2)).subs(x ** S.Half, L) == x ** Rational(-1, 2)
    assert (x ** Rational(-1, 2)).subs(x ** Rational(-1, 2), L) == L
    assert (x ** (2 * someint)).subs(x ** someint, L) == L ** 2
    assert (x ** (2 * someint + 3)).subs(x ** someint, L) == L ** 2 * x ** 3
    assert (x ** (3 * someint + 3)).subs(x ** someint, L) == L ** 3 * x ** 3
    assert (x ** (3 * someint)).subs(x ** (2 * someint), L) == L * x ** someint
    assert (x ** (4 * someint)).subs(x ** (2 * someint), L) == L ** 2
    assert (x ** (4 * someint + 1)).subs(x ** (2 * someint), L) == L ** 2 * x
    assert (x ** (4 * someint)).subs(x ** (3 * someint), L) == L * x ** someint
    assert (x ** (4 * someint + 1)).subs(x ** (3 * someint), L) == L * x ** (someint + 1)
    assert (x ** (2 * alpha)).subs(x ** alpha, L) == x ** (2 * alpha)
    assert (x ** (2 * alpha + 2)).subs(x ** 2, L) == x ** (2 * alpha + 2)
    assert ((2 * z) ** alpha).subs(z ** alpha, y) == (2 * z) ** alpha
    assert (x ** (2 * someint * alpha)).subs(x ** someint, L) == x ** (2 * someint * alpha)
    assert (x ** (2 * someint + alpha)).subs(x ** someint, L) == x ** (2 * someint + alpha)
    assert (x ** (someint ** 2 + 3)).subs(x ** someint, L) == x ** (someint ** 2 + 3)
    assert (4 ** z).subs(2 ** z, y) == y ** 2
    assert (x ** (-1)).subs(x ** 3, L) == x ** (-1)
    assert (x ** (-2)).subs(x ** 3, L) == x ** (-2)
    assert (x ** (-3)).subs(x ** 3, L) == L ** (-1)
    assert (x ** (-4)).subs(x ** 3, L) == L ** (-1) * x ** (-1)
    assert (x ** (-5)).subs(x ** 3, L) == L ** (-1) * x ** (-2)
    assert (x ** (-1)).subs(x ** (-3), L) == x ** (-1)
    assert (x ** (-2)).subs(x ** (-3), L) == x ** (-2)
    assert (x ** (-3)).subs(x ** (-3), L) == L
    assert (x ** (-4)).subs(x ** (-3), L) == L * x ** (-1)
    assert (x ** (-5)).subs(x ** (-3), L) == L * x ** (-2)
    assert (x ** 1).subs(x ** (-3), L) == x
    assert (x ** 2).subs(x ** (-3), L) == x ** 2
    assert (x ** 3).subs(x ** (-3), L) == L ** (-1)
    assert (x ** 4).subs(x ** (-3), L) == L ** (-1) * x
    assert (x ** 5).subs(x ** (-3), L) == L ** (-1) * x ** 2