from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
def test_riccati_reduced():
    """
    This function tests the transformation of a
    Riccati ODE to its normal Riccati ODE.

    Each test case 2 values -

    1. eq - A Riccati ODE.
    2. normal_eq - The normal Riccati ODE of eq.
    """
    tests = [(f(x).diff(x) - x ** 2 - x * f(x) - x * f(x) ** 2, f(x).diff(x) + f(x) ** 2 + x ** 3 - x ** 2 / 4 - 3 / (4 * x ** 2)), (6 * x / (2 * x + 9) + f(x).diff(x) - (x + 1) * f(x) ** 2 / x, -3 * x ** 2 * (1 / x + (-x - 1) / x ** 2) ** 2 / (4 * (-x - 1) ** 2) + Mul(6, -x - 1, evaluate=False) / (2 * x + 9) + f(x) ** 2 + f(x).diff(x) - (-1 + (x + 1) / x) / (x * (-x - 1))), (f(x) ** 2 + f(x).diff(x) - (x - 1) * f(x) / (-x - S(1) / 2), -(2 * x - 2) ** 2 / (4 * (2 * x + 1) ** 2) + (2 * x - 2) / (2 * x + 1) ** 2 + f(x) ** 2 + f(x).diff(x) - 1 / (2 * x + 1)), (f(x).diff(x) - f(x) ** 2 / x, f(x) ** 2 + f(x).diff(x) + 1 / (4 * x ** 2)), (-3 * (-x ** 2 - x + 1) / (x ** 2 + 6 * x + 1) + f(x).diff(x) + f(x) ** 2 / x, f(x) ** 2 + f(x).diff(x) + (3 * x ** 2 / (x ** 2 + 6 * x + 1) + 3 * x / (x ** 2 + 6 * x + 1) - 3 / (x ** 2 + 6 * x + 1)) / x + 1 / (4 * x ** 2)), (6 * x / (2 * x + 9) + f(x).diff(x) - (x + 1) * f(x) / x, False), (f(x) * f(x).diff(x) - 1 / x + f(x) / 3 + f(x) ** 2 / (x ** 2 - 2), False)]
    for eq, normal_eq in tests:
        assert normal_eq == riccati_reduced(eq, f, x)