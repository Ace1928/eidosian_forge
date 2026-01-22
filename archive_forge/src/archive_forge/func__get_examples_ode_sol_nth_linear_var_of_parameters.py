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
def _get_examples_ode_sol_nth_linear_var_of_parameters():
    g = exp(-x)
    f2 = f(x).diff(x, 2)
    c = 3 * f(x).diff(x, 3) + 5 * f2 + f(x).diff(x) - f(x) - x
    return {'hint': 'nth_linear_constant_coeff_variation_of_parameters', 'func': f(x), 'examples': {'var_of_parameters_01': {'eq': c - x * g, 'sol': [Eq(f(x), C3 * exp(x / 3) - x + (C1 + x * (C2 - x ** 2 / 24 - 3 * x / 32)) * exp(-x) - 1)], 'slow': True}, 'var_of_parameters_02': {'eq': c - g, 'sol': [Eq(f(x), C3 * exp(x / 3) - x + (C1 + x * (C2 - x / 8)) * exp(-x) - 1)], 'slow': True}, 'var_of_parameters_03': {'eq': f(x).diff(x) - 1, 'sol': [Eq(f(x), C1 + x)], 'slow': True}, 'var_of_parameters_04': {'eq': f2 + 3 * f(x).diff(x) + 2 * f(x) - 4, 'sol': [Eq(f(x), C1 * exp(-2 * x) + C2 * exp(-x) + 2)], 'slow': True}, 'var_of_parameters_05': {'eq': f2 + 3 * f(x).diff(x) + 2 * f(x) - 12 * exp(x), 'sol': [Eq(f(x), C1 * exp(-2 * x) + C2 * exp(-x) + 2 * exp(x))], 'slow': True}, 'var_of_parameters_06': {'eq': f2 - 2 * f(x).diff(x) - 8 * f(x) - 9 * x * exp(x) - 10 * exp(-x), 'sol': [Eq(f(x), -x * exp(x) - 2 * exp(-x) + C1 * exp(-2 * x) + C2 * exp(4 * x))], 'slow': True}, 'var_of_parameters_07': {'eq': f2 + 2 * f(x).diff(x) + f(x) - x ** 2 * exp(-x), 'sol': [Eq(f(x), (C1 + x * (C2 + x ** 3 / 12)) * exp(-x))], 'slow': True}, 'var_of_parameters_08': {'eq': f2 - 3 * f(x).diff(x) + 2 * f(x) - x * exp(-x), 'sol': [Eq(f(x), C1 * exp(x) + C2 * exp(2 * x) + (6 * x + 5) * exp(-x) / 36)], 'slow': True}, 'var_of_parameters_09': {'eq': f(x).diff(x, 3) - 3 * f2 + 3 * f(x).diff(x) - f(x) - exp(x), 'sol': [Eq(f(x), (C1 + x * (C2 + x * (C3 + x / 6))) * exp(x))], 'slow': True}, 'var_of_parameters_10': {'eq': f2 + 2 * f(x).diff(x) + f(x) - exp(-x) / x, 'sol': [Eq(f(x), (C1 + x * (C2 + log(x))) * exp(-x))], 'slow': True}, 'var_of_parameters_11': {'eq': f2 + f(x) - 1 / sin(x) * 1 / cos(x), 'sol': [Eq(f(x), (C1 + log(sin(x) - 1) / 2 - log(sin(x) + 1) / 2) * cos(x) + (C2 + log(cos(x) - 1) / 2 - log(cos(x) + 1) / 2) * sin(x))], 'slow': True}, 'var_of_parameters_12': {'eq': f(x).diff(x, 4) - 1 / x, 'sol': [Eq(f(x), C1 + C2 * x + C3 * x ** 2 + x ** 3 * (C4 + log(x) / 6))], 'slow': True}, 'var_of_parameters_13': {'eq': f(x).diff(x, 5) + 2 * f(x).diff(x, 3) + f(x).diff(x) - 2 * x - exp(I * x), 'sol': [Eq(f(x), C1 + x ** 2 + (C2 + x * (C3 - x / 8 + 3 * exp(I * x) / 2 + 3 * exp(-I * x) / 2) + 5 * exp(2 * I * x) / 16 + 2 * I * exp(I * x) - 2 * I * exp(-I * x)) * sin(x) + (C4 + x * (C5 + I * x / 8 + 3 * I * exp(I * x) / 2 - 3 * I * exp(-I * x) / 2) + 5 * I * exp(2 * I * x) / 16 - 2 * exp(I * x) - 2 * exp(-I * x)) * cos(x) - I * exp(I * x))]}, 'var_of_parameters_14': {'eq': f(x).diff(x, 5) + 2 * f(x).diff(x, 3) + f(x).diff(x) - exp(I * x), 'sol': [Eq(f(x), C1 + (C2 + x * (C3 - x / 8) + 5 * exp(2 * I * x) / 16) * sin(x) + (C4 + x * (C5 + I * x / 8) + 5 * I * exp(2 * I * x) / 16) * cos(x) - I * exp(I * x))]}, 'var_of_parameters_15': {'eq': Derivative(f(x), x, x) + 9 * f(x) - sec(x), 'sol': [Eq(f(x), (C1 - x / 3 + sin(2 * x) / 3) * sin(3 * x) + (C2 + log(cos(x)) - 2 * log(cos(x) ** 2) / 3 + 2 * cos(x) ** 2 / 3) * cos(3 * x))], 'slow': True}}}