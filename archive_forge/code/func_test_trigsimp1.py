from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
def test_trigsimp1():
    x, y = symbols('x,y')
    assert trigsimp(1 - sin(x) ** 2) == cos(x) ** 2
    assert trigsimp(1 - cos(x) ** 2) == sin(x) ** 2
    assert trigsimp(sin(x) ** 2 + cos(x) ** 2) == 1
    assert trigsimp(1 + tan(x) ** 2) == 1 / cos(x) ** 2
    assert trigsimp(1 / cos(x) ** 2 - 1) == tan(x) ** 2
    assert trigsimp(1 / cos(x) ** 2 - tan(x) ** 2) == 1
    assert trigsimp(1 + cot(x) ** 2) == 1 / sin(x) ** 2
    assert trigsimp(1 / sin(x) ** 2 - 1) == 1 / tan(x) ** 2
    assert trigsimp(1 / sin(x) ** 2 - cot(x) ** 2) == 1
    assert trigsimp(5 * cos(x) ** 2 + 5 * sin(x) ** 2) == 5
    assert trigsimp(5 * cos(x / 2) ** 2 + 2 * sin(x / 2) ** 2) == 3 * cos(x) / 2 + Rational(7, 2)
    assert trigsimp(sin(x) / cos(x)) == tan(x)
    assert trigsimp(2 * tan(x) * cos(x)) == 2 * sin(x)
    assert trigsimp(cot(x) ** 3 * sin(x) ** 3) == cos(x) ** 3
    assert trigsimp(y * tan(x) ** 2 / sin(x) ** 2) == y / cos(x) ** 2
    assert trigsimp(cot(x) / cos(x)) == 1 / sin(x)
    assert trigsimp(sin(x + y) + sin(x - y)) == 2 * sin(x) * cos(y)
    assert trigsimp(sin(x + y) - sin(x - y)) == 2 * sin(y) * cos(x)
    assert trigsimp(cos(x + y) + cos(x - y)) == 2 * cos(x) * cos(y)
    assert trigsimp(cos(x + y) - cos(x - y)) == -2 * sin(x) * sin(y)
    assert trigsimp(tan(x + y) - tan(x) / (1 - tan(x) * tan(y))) == sin(y) / (-sin(y) * tan(x) + cos(y))
    assert trigsimp(sinh(x + y) + sinh(x - y)) == 2 * sinh(x) * cosh(y)
    assert trigsimp(sinh(x + y) - sinh(x - y)) == 2 * sinh(y) * cosh(x)
    assert trigsimp(cosh(x + y) + cosh(x - y)) == 2 * cosh(x) * cosh(y)
    assert trigsimp(cosh(x + y) - cosh(x - y)) == 2 * sinh(x) * sinh(y)
    assert trigsimp(tanh(x + y) - tanh(x) / (1 + tanh(x) * tanh(y))) == sinh(y) / (sinh(y) * tanh(x) + cosh(y))
    assert trigsimp(cos(0.12345) ** 2 + sin(0.12345) ** 2) == 1.0
    e = 2 * sin(x) ** 2 + 2 * cos(x) ** 2
    assert trigsimp(log(e)) == log(2)