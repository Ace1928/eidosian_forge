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
def test_trigsimp1a():
    assert trigsimp(sin(2) ** 2 * cos(3) * exp(2) / cos(2) ** 2) == tan(2) ** 2 * cos(3) * exp(2)
    assert trigsimp(tan(2) ** 2 * cos(3) * exp(2) * cos(2) ** 2) == sin(2) ** 2 * cos(3) * exp(2)
    assert trigsimp(cot(2) * cos(3) * exp(2) * sin(2)) == cos(3) * exp(2) * cos(2)
    assert trigsimp(tan(2) * cos(3) * exp(2) / sin(2)) == cos(3) * exp(2) / cos(2)
    assert trigsimp(cot(2) * cos(3) * exp(2) / cos(2)) == cos(3) * exp(2) / sin(2)
    assert trigsimp(cot(2) * cos(3) * exp(2) * tan(2)) == cos(3) * exp(2)
    assert trigsimp(sinh(2) * cos(3) * exp(2) / cosh(2)) == tanh(2) * cos(3) * exp(2)
    assert trigsimp(tanh(2) * cos(3) * exp(2) * cosh(2)) == sinh(2) * cos(3) * exp(2)
    assert trigsimp(coth(2) * cos(3) * exp(2) * sinh(2)) == cosh(2) * cos(3) * exp(2)
    assert trigsimp(tanh(2) * cos(3) * exp(2) / sinh(2)) == cos(3) * exp(2) / cosh(2)
    assert trigsimp(coth(2) * cos(3) * exp(2) / cosh(2)) == cos(3) * exp(2) / sinh(2)
    assert trigsimp(coth(2) * cos(3) * exp(2) * tanh(2)) == cos(3) * exp(2)