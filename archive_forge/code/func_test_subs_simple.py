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
def test_subs_simple():
    a = symbols('a', commutative=True)
    x = symbols('x', commutative=False)
    assert (2 * a).subs(1, 3) == 2 * a
    assert (2 * a).subs(2, 3) == 3 * a
    assert (2 * a).subs(a, 3) == 6
    assert sin(2).subs(1, 3) == sin(2)
    assert sin(2).subs(2, 3) == sin(3)
    assert sin(a).subs(a, 3) == sin(3)
    assert (2 * x).subs(1, 3) == 2 * x
    assert (2 * x).subs(2, 3) == 3 * x
    assert (2 * x).subs(x, 3) == 6
    assert sin(x).subs(x, 3) == sin(3)