from sympy.core.function import Function
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.ode import constantsimp, constant_renumber
from sympy.testing.pytest import XFAIL
def test_ode_solutions():
    assert constant_renumber(constantsimp(C1 * exp(2 * x) + exp(x) * (C2 + C3), [C1, C2, C3])) == constant_renumber(C1 * exp(x) + C2 * exp(2 * x))
    assert constant_renumber(constantsimp(Eq(f(x), I * C1 * sinh(x / 3) + C2 * cosh(x / 3)), [C1, C2])) == constant_renumber(Eq(f(x), C1 * sinh(x / 3) + C2 * cosh(x / 3)))
    assert constant_renumber(constantsimp(Eq(f(x), acos(-C1 / cos(x))), [C1])) == Eq(f(x), acos(C1 / cos(x)))
    assert constant_renumber(constantsimp(Eq(log(f(x) / C1) + 2 * exp(x / f(x)), 0), [C1])) == Eq(log(C1 * f(x)) + 2 * exp(x / f(x)), 0)
    assert constant_renumber(constantsimp(Eq(log(x * sqrt(2) * sqrt(1 / x) * sqrt(f(x)) / C1) + x ** 2 / (2 * f(x) ** 2), 0), [C1])) == Eq(log(C1 * sqrt(x) * sqrt(f(x))) + x ** 2 / (2 * f(x) ** 2), 0)
    assert constant_renumber(constantsimp(Eq(-exp(-f(x) / x) * sin(f(x) / x) / 2 + log(x / C1) - cos(f(x) / x) * exp(-f(x) / x) / 2, 0), [C1])) == Eq(-exp(-f(x) / x) * sin(f(x) / x) / 2 + log(C1 * x) - cos(f(x) / x) * exp(-f(x) / x) / 2, 0)
    assert constant_renumber(constantsimp(Eq(-Integral(-1 / (sqrt(1 - u2 ** 2) * u2), (u2, _a, x / f(x))) + log(f(x) / C1), 0), [C1])) == Eq(-Integral(-1 / (u2 * sqrt(1 - u2 ** 2)), (u2, _a, x / f(x))) + log(C1 * f(x)), 0)
    assert [constantsimp(i, [C1]) for i in [Eq(f(x), sqrt(-C1 * x + x ** 2)), Eq(f(x), -sqrt(-C1 * x + x ** 2))]] == [Eq(f(x), sqrt(x * (C1 + x))), Eq(f(x), -sqrt(x * (C1 + x)))]