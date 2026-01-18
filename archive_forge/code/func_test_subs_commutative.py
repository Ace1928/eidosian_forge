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
def test_subs_commutative():
    a, b, c, d, K = symbols('a b c d K', commutative=True)
    assert (a * b).subs(a * b, K) == K
    assert (a * b * a * b).subs(a * b, K) == K ** 2
    assert (a * a * b * b).subs(a * b, K) == K ** 2
    assert (a * b * c * d).subs(a * b * c, K) == d * K
    assert (a * b ** c).subs(a, K) == K * b ** c
    assert (a * b ** c).subs(b, K) == a * K ** c
    assert (a * b ** c).subs(c, K) == a * b ** K
    assert (a * b * c * b * a).subs(a * b, K) == c * K ** 2
    assert (a ** 3 * b ** 2 * a).subs(a * b, K) == a ** 2 * K ** 2