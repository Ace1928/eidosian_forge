from sympy.core.function import (Derivative, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (Ei, erfi)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.rootoftools import rootof
from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import raises, slow, ON_CI
import traceback
from sympy.solvers.ode.tests.test_single import _test_an_example
@_add_example_keys
def _get_examples_ode_sol_separable():
    t, a = symbols('a,t')
    m = 96
    g = 9.8
    k = 0.2
    f1 = g * m
    v = Function('v')
    return {'hint': 'separable', 'func': f(x), 'examples': {'separable_01': {'eq': f(x).diff(x) - f(x), 'sol': [Eq(f(x), C1 * exp(x))]}, 'separable_02': {'eq': x * f(x).diff(x) - f(x), 'sol': [Eq(f(x), C1 * x)]}, 'separable_03': {'eq': f(x).diff(x) + sin(x), 'sol': [Eq(f(x), C1 + cos(x))]}, 'separable_04': {'eq': f(x) ** 2 + 1 - (x ** 2 + 1) * f(x).diff(x), 'sol': [Eq(f(x), tan(C1 + atan(x)))]}, 'separable_05': {'eq': f(x).diff(x) / tan(x) - f(x) - 2, 'sol': [Eq(f(x), C1 / cos(x) - 2)]}, 'separable_06': {'eq': f(x).diff(x) * (1 - sin(f(x))) - 1, 'sol': [Eq(-x + f(x) + cos(f(x)), C1)]}, 'separable_07': {'eq': f(x) * x ** 2 * f(x).diff(x) - f(x) ** 3 - 2 * x ** 2 * f(x).diff(x), 'sol': [Eq(f(x), (-x - sqrt(x * (4 * C1 * x + x - 4))) / (C1 * x - 1) / 2), Eq(f(x), (-x + sqrt(x * (4 * C1 * x + x - 4))) / (C1 * x - 1) / 2)], 'slow': True}, 'separable_08': {'eq': f(x) ** 2 - 1 - (2 * f(x) + x * f(x)) * f(x).diff(x), 'sol': [Eq(f(x), -sqrt(C1 * x ** 2 + 4 * C1 * x + 4 * C1 + 1)), Eq(f(x), sqrt(C1 * x ** 2 + 4 * C1 * x + 4 * C1 + 1))], 'slow': True}, 'separable_09': {'eq': x * log(x) * f(x).diff(x) + sqrt(1 + f(x) ** 2), 'sol': [Eq(f(x), sinh(C1 - log(log(x))))], 'slow': True, 'checkodesol_XFAIL': True}, 'separable_10': {'eq': exp(x + 1) * tan(f(x)) + cos(f(x)) * f(x).diff(x), 'sol': [Eq(E * exp(x) + log(cos(f(x)) - 1) / 2 - log(cos(f(x)) + 1) / 2 + cos(f(x)), C1)], 'slow': True}, 'separable_11': {'eq': x * cos(f(x)) + x ** 2 * sin(f(x)) * f(x).diff(x) - a ** 2 * sin(f(x)) * f(x).diff(x), 'sol': [Eq(f(x), -acos(C1 * sqrt(-a ** 2 + x ** 2)) + 2 * pi), Eq(f(x), acos(C1 * sqrt(-a ** 2 + x ** 2)))], 'slow': True}, 'separable_12': {'eq': f(x).diff(x) - f(x) * tan(x), 'sol': [Eq(f(x), C1 / cos(x))]}, 'separable_13': {'eq': (x - 1) * cos(f(x)) * f(x).diff(x) - 2 * x * sin(f(x)), 'sol': [Eq(f(x), pi - asin(C1 * (x ** 2 - 2 * x + 1) * exp(2 * x))), Eq(f(x), asin(C1 * (x ** 2 - 2 * x + 1) * exp(2 * x)))]}, 'separable_14': {'eq': f(x).diff(x) - f(x) * log(f(x)) / tan(x), 'sol': [Eq(f(x), exp(C1 * sin(x)))]}, 'separable_15': {'eq': x * f(x).diff(x) + (1 + f(x) ** 2) * atan(f(x)), 'sol': [Eq(f(x), tan(C1 / x))], 'slow': True, 'checkodesol_XFAIL': True}, 'separable_16': {'eq': f(x).diff(x) + x * (f(x) + 1), 'sol': [Eq(f(x), -1 + C1 * exp(-x ** 2 / 2))]}, 'separable_17': {'eq': exp(f(x) ** 2) * (x ** 2 + 2 * x + 1) + (x * f(x) + f(x)) * f(x).diff(x), 'sol': [Eq(f(x), -sqrt(log(1 / (C1 + x ** 2 + 2 * x)))), Eq(f(x), sqrt(log(1 / (C1 + x ** 2 + 2 * x))))]}, 'separable_18': {'eq': f(x).diff(x) + f(x), 'sol': [Eq(f(x), C1 * exp(-x))]}, 'separable_19': {'eq': sin(x) * cos(2 * f(x)) + cos(x) * sin(2 * f(x)) * f(x).diff(x), 'sol': [Eq(f(x), pi - acos(C1 / cos(x) ** 2) / 2), Eq(f(x), acos(C1 / cos(x) ** 2) / 2)]}, 'separable_20': {'eq': (1 - x) * f(x).diff(x) - x * (f(x) + 1), 'sol': [Eq(f(x), (C1 * exp(-x) - x + 1) / (x - 1))]}, 'separable_21': {'eq': f(x) * diff(f(x), x) + x - 3 * x * f(x) ** 2, 'sol': [Eq(f(x), -sqrt(3) * sqrt(C1 * exp(3 * x ** 2) + 1) / 3), Eq(f(x), sqrt(3) * sqrt(C1 * exp(3 * x ** 2) + 1) / 3)]}, 'separable_22': {'eq': f(x).diff(x) - exp(x + f(x)), 'sol': [Eq(f(x), log(-1 / (C1 + exp(x))))], 'XFAIL': ['lie_group']}, 'separable_23': {'eq': x * f(x).diff(x) + 1 - f(x) ** 2, 'sol': [Eq(f(x), (-C1 - x ** 2) / (-C1 + x ** 2))]}, 'separable_24': {'eq': f(t).diff(t) - (1 - 51.05 * y * f(t)), 'sol': [Eq(f(t), (0.019588638589618023 * exp(y * (C1 - 51.05 * t)) + 0.019588638589618023) / y)], 'func': f(t)}, 'separable_25': {'eq': f(x).diff(x) - C1 * f(x), 'sol': [Eq(f(x), C2 * exp(C1 * x))]}, 'separable_26': {'eq': f1 - k * v(t) ** 2 - m * Derivative(v(t)), 'sol': [Eq(v(t), -68.58571279792899 / tanh(C1 - 0.14288690166235204 * t))], 'func': v(t), 'checkodesol_XFAIL': True}, 'separable_27': {'eq': f(x).diff(x) - exp(f(x) - x), 'sol': [Eq(f(x), log(-exp(x) / (C1 * exp(x) - 1)))]}}}