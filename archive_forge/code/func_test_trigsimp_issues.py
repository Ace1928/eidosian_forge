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
def test_trigsimp_issues():
    a, x, y = symbols('a x y')
    assert trigsimp(sin(x) ** 3 + cos(x) ** 2 * sin(x)) == sin(x)
    assert trigsimp(diff(integrate(cos(x) / sin(x) ** 3, x), x)) == cos(x) / sin(x) ** 3
    assert trigsimp(diff(integrate(sin(x) / cos(x) ** 3, x), x)) == sin(x) / cos(x) ** 3
    e = sin(x) ** y / cos(x) ** y
    assert trigsimp(e) == e
    assert trigsimp(e.subs(y, 2)) == tan(x) ** 2
    assert trigsimp(e.subs(x, 1)) == tan(1) ** y
    assert (cos(x) ** 2 / sin(x) ** 2 * cos(y) ** 2 / sin(y) ** 2).trigsimp() == 1 / tan(x) ** 2 / tan(y) ** 2
    assert trigsimp(cos(x) / sin(x) * cos(x + y) / sin(x + y)) == 1 / (tan(x) * tan(x + y))
    eq = cos(2) * (cos(3) + 1) ** 2 / (cos(3) - 1) ** 2
    assert trigsimp(eq) == eq.factor()
    assert trigsimp(cos(2) * (cos(3) + 1) ** 2 * (cos(3) - 1) ** 2) == cos(2) * sin(3) ** 4
    assert cot(x).equals(tan(x)) is False
    z = cos(x) ** 2 + sin(x) ** 2 - 1
    z1 = tan(x) ** 2 - 1 / cot(x) ** 2
    n = 1 + z1 / z
    assert trigsimp(sin(n)) != sin(1)
    eq = x * (n - 1) - x * n
    assert trigsimp(eq) is S.NaN
    assert trigsimp(eq, recursive=True) is S.NaN
    assert trigsimp(1).is_Integer
    assert trigsimp(-sin(x) ** 4 - 2 * sin(x) ** 2 * cos(x) ** 2 - cos(x) ** 4) == -1