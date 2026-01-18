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
def test_subs_dict():
    a, b, c, d, e = symbols('a b c d e')
    assert (2 * x + y + z).subs({'x': 1, 'y': 2}) == 4 + z
    l = [(sin(x), 2), (x, 1)]
    assert sin(x).subs(l) == sin(x).subs(dict(l)) == 2
    assert sin(x).subs(reversed(l)) == sin(1)
    expr = sin(2 * x) + sqrt(sin(2 * x)) * cos(2 * x) * sin(exp(x) * x)
    reps = {sin(2 * x): c, sqrt(sin(2 * x)): a, cos(2 * x): b, exp(x): e, x: d}
    assert expr.subs(reps) == c + a * b * sin(d * e)
    l = [(x, 3), (y, x ** 2)]
    assert (x + y).subs(l) == 3 + x ** 2
    assert (x + y).subs(reversed(l)) == 12
    l = [(y, z + 2), (1 + z, 5), (z, 2)]
    assert (y - 1 + 3 * x).subs(l) == 5 + 3 * x
    l = [(y, z + 2), (z, 3)]
    assert (y - 2).subs(l) == 3