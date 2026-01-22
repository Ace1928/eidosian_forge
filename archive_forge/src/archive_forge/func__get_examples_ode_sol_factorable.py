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
def _get_examples_ode_sol_factorable():
    """ some hints are marked as xfail for examples because they missed additional algebraic solution
    which could be found by Factorable hint. Fact_01 raise exception for
    nth_linear_constant_coeff_undetermined_coefficients"""
    y = Dummy('y')
    a0, a1, a2, a3, a4 = symbols('a0, a1, a2, a3, a4')
    return {'hint': 'factorable', 'func': f(x), 'examples': {'fact_01': {'eq': f(x) + f(x) * f(x).diff(x), 'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)], 'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best', '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep', 'lie_group', 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients', 'nth_linear_constant_coeff_variation_of_parameters', 'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters', 'nth_linear_constant_coeff_undetermined_coefficients']}, 'fact_02': {'eq': f(x) * (f(x).diff(x) + f(x) * x + 2), 'sol': [Eq(f(x), (C1 - sqrt(2) * sqrt(pi) * erfi(sqrt(2) * x / 2)) * exp(-x ** 2 / 2)), Eq(f(x), 0)], 'XFAIL': ['Bernoulli', '1st_linear', 'lie_group']}, 'fact_03': {'eq': (f(x).diff(x) + f(x) * x ** 2) * (f(x).diff(x, 2) + x * f(x)), 'sol': [Eq(f(x), C1 * airyai(-x) + C2 * airybi(-x)), Eq(f(x), C1 * exp(-x ** 3 / 3))]}, 'fact_04': {'eq': (f(x).diff(x) + f(x) * x ** 2) * (f(x).diff(x, 2) + f(x)), 'sol': [Eq(f(x), C1 * exp(-x ** 3 / 3)), Eq(f(x), C1 * sin(x) + C2 * cos(x))]}, 'fact_05': {'eq': (f(x).diff(x) ** 2 - 1) * (f(x).diff(x) ** 2 - 4), 'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x), Eq(f(x), C1 + 2 * x), Eq(f(x), C1 - 2 * x)]}, 'fact_06': {'eq': (f(x).diff(x, 2) - exp(f(x))) * f(x).diff(x), 'sol': [Eq(f(x), log(-C1 / (cos(sqrt(-C1) * (C2 + x)) + 1))), Eq(f(x), log(-C1 / (cos(sqrt(-C1) * (C2 - x)) + 1))), Eq(f(x), C1)], 'slow': True}, 'fact_07': {'eq': (f(x).diff(x) ** 2 - 1) * (f(x) * f(x).diff(x) - 1), 'sol': [Eq(f(x), C1 - x), Eq(f(x), -sqrt(C1 + 2 * x)), Eq(f(x), sqrt(C1 + 2 * x)), Eq(f(x), C1 + x)]}, 'fact_08': {'eq': Derivative(f(x), x) ** 4 - 2 * Derivative(f(x), x) ** 2 + 1, 'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x)]}, 'fact_09': {'eq': f(x) ** 2 * Derivative(f(x), x) ** 6 - 2 * f(x) ** 2 * Derivative(f(x), x) ** 4 + f(x) ** 2 * Derivative(f(x), x) ** 2 - 2 * f(x) * Derivative(f(x), x) ** 5 + 4 * f(x) * Derivative(f(x), x) ** 3 - 2 * f(x) * Derivative(f(x), x) + Derivative(f(x), x) ** 4 - 2 * Derivative(f(x), x) ** 2 + 1, 'sol': [Eq(f(x), C1 - x), Eq(f(x), -sqrt(C1 + 2 * x)), Eq(f(x), sqrt(C1 + 2 * x)), Eq(f(x), C1 + x)]}, 'fact_10': {'eq': x ** 4 * f(x) ** 2 + 2 * x ** 4 * f(x) * Derivative(f(x), (x, 2)) + x ** 4 * Derivative(f(x), (x, 2)) ** 2 + 2 * x ** 3 * f(x) * Derivative(f(x), x) + 2 * x ** 3 * Derivative(f(x), x) * Derivative(f(x), (x, 2)) - 7 * x ** 2 * f(x) ** 2 - 7 * x ** 2 * f(x) * Derivative(f(x), (x, 2)) + x ** 2 * Derivative(f(x), x) ** 2 - 7 * x * f(x) * Derivative(f(x), x) + 12 * f(x) ** 2, 'sol': [Eq(f(x), C1 * besselj(2, x) + C2 * bessely(2, x)), Eq(f(x), C1 * besselj(sqrt(3), x) + C2 * bessely(sqrt(3), x))], 'slow': True}, 'fact_11': {'eq': (f(x).diff(x, 2) - exp(f(x))) * (f(x).diff(x, 2) + exp(f(x))), 'sol': [Eq(f(x), log(C1 / (cos(C1 * sqrt(-1 / C1) * (C2 + x)) - 1))), Eq(f(x), log(C1 / (cos(C1 * sqrt(-1 / C1) * (C2 - x)) - 1))), Eq(f(x), log(C1 / (1 - cos(C1 * sqrt(-1 / C1) * (C2 + x))))), Eq(f(x), log(C1 / (1 - cos(C1 * sqrt(-1 / C1) * (C2 - x)))))], 'dsolve_too_slow': True}, 'fact_12': {'eq': exp(f(x).diff(x)) - f(x) ** 2, 'sol': [Eq(NonElementaryIntegral(1 / log(y ** 2), (y, f(x))), C1 + x)], 'XFAIL': ['lie_group']}, 'fact_13': {'eq': f(x).diff(x) ** 2 - f(x) ** 3, 'sol': [Eq(f(x), 4 / (C1 ** 2 - 2 * C1 * x + x ** 2))], 'XFAIL': ['lie_group']}, 'fact_14': {'eq': f(x).diff(x) ** 2 - f(x), 'sol': [Eq(f(x), C1 ** 2 / 4 - C1 * x / 2 + x ** 2 / 4)]}, 'fact_15': {'eq': f(x).diff(x) ** 2 - f(x) ** 2, 'sol': [Eq(f(x), C1 * exp(x)), Eq(f(x), C1 * exp(-x))]}, 'fact_16': {'eq': f(x).diff(x) ** 2 - f(x) ** 3, 'sol': [Eq(f(x), 4 / (C1 ** 2 - 2 * C1 * x + x ** 2))]}, 'fact_17': {'eq': f(x).diff(x) - (a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0) ** (-1 / 2), 'sol': [Eq(f(x), C1 + Integral(1 / sqrt(a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4), x))], 'slow': True}, 'fact_18': {'eq': Eq(f(2 * x), sin(Derivative(f(x)))), 'sol': [Eq(f(x), C1 + Integral(pi - asin(f(2 * x)), x)), Eq(f(x), C1 + Integral(asin(f(2 * x)), x))], 'checkodesol_XFAIL': True}, 'fact_19': {'eq': Derivative(f(x), x) ** 2 - x ** 3, 'sol': [Eq(f(x), C1 - 2 * x ** Rational(5, 2) / 5), Eq(f(x), C1 + 2 * x ** Rational(5, 2) / 5)]}, 'fact_20': {'eq': x * f(x).diff(x, 2) - x * f(x), 'sol': [Eq(f(x), C1 * exp(-x) + C2 * exp(x))]}}}