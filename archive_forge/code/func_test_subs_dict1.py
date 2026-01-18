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
def test_subs_dict1():
    assert (1 + x * y).subs(x, pi) == 1 + pi * y
    assert (1 + x * y).subs({x: pi, y: 2}) == 1 + 2 * pi
    c2, c3, q1p, q2p, c1, s1, s2, s3 = symbols('c2 c3 q1p q2p c1 s1 s2 s3')
    test = c2 ** 2 * q2p * c3 + c1 ** 2 * s2 ** 2 * q2p * c3 + s1 ** 2 * s2 ** 2 * q2p * c3 - c1 ** 2 * q1p * c2 * s3 - s1 ** 2 * q1p * c2 * s3
    assert test.subs({c1 ** 2: 1 - s1 ** 2, c2 ** 2: 1 - s2 ** 2, c3 ** 3: 1 - s3 ** 2}) == c3 * q2p * (1 - s2 ** 2) + c3 * q2p * s2 ** 2 * (1 - s1 ** 2) - c2 * q1p * s3 * (1 - s1 ** 2) + c3 * q2p * s1 ** 2 * s2 ** 2 - c2 * q1p * s3 * s1 ** 2