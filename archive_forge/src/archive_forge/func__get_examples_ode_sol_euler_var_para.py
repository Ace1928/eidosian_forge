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
def _get_examples_ode_sol_euler_var_para():
    return {'hint': 'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters', 'func': f(x), 'examples': {'euler_var_01': {'eq': Eq(x ** 2 * Derivative(f(x), x, x) - 2 * x * Derivative(f(x), x) + 2 * f(x), x ** 4), 'sol': [Eq(f(x), x * (C1 + C2 * x + x ** 3 / 6))]}, 'euler_var_02': {'eq': Eq(3 * x ** 2 * diff(f(x), x, x) + 6 * x * diff(f(x), x) - 6 * f(x), x ** 3 * exp(x)), 'sol': [Eq(f(x), C1 / x ** 2 + C2 * x + x * exp(x) / 3 - 4 * exp(x) / 3 + 8 * exp(x) / (3 * x) - 8 * exp(x) / (3 * x ** 2))]}, 'euler_var_03': {'eq': Eq(x ** 2 * Derivative(f(x), x, x) - 2 * x * Derivative(f(x), x) + 2 * f(x), x ** 4 * exp(x)), 'sol': [Eq(f(x), x * (C1 + C2 * x + x * exp(x) - 2 * exp(x)))]}, 'euler_var_04': {'eq': x ** 2 * Derivative(f(x), x, x) - 2 * x * Derivative(f(x), x) + 2 * f(x) - log(x), 'sol': [Eq(f(x), C1 * x + C2 * x ** 2 + log(x) / 2 + Rational(3, 4))]}, 'euler_var_05': {'eq': -exp(x) + (x * Derivative(f(x), (x, 2)) + Derivative(f(x), x)) / x, 'sol': [Eq(f(x), C1 + C2 * log(x) + exp(x) - Ei(x))]}, 'euler_var_06': {'eq': x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + 4 * f(x) - 1 / x, 'sol': [Eq(f(x), C1 * sin(2 * log(x)) + C2 * cos(2 * log(x)) + 1 / (5 * x))]}}}