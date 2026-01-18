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
def test_subs_basic_funcs():
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    w, x, y, z, L = symbols('w x y z L', commutative=False)
    assert (x + y).subs(x + y, L) == L
    assert (x - y).subs(x - y, L) == L
    assert (x / y).subs(x, L) == L / y
    assert (x ** y).subs(x, L) == L ** y
    assert (x ** y).subs(y, L) == x ** L
    assert ((a - c) / b).subs(b, K) == (a - c) / K
    assert exp(x * y - z).subs(x * y, L) == exp(L - z)
    assert (a * exp(x * y - w * z) + b * exp(x * y + w * z)).subs(z, 0) == a * exp(x * y) + b * exp(x * y)
    assert ((a - b) / (c * d - a * b)).subs(c * d - a * b, K) == (a - b) / K
    assert (w * exp(a * b - c) * x * y / 4).subs(x * y, L) == w * exp(a * b - c) * L / 4