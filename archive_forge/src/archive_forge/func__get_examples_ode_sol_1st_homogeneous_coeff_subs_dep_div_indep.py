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
def _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep():
    return {'hint': '1st_homogeneous_coeff_subs_dep_div_indep', 'func': f(x), 'examples': {'dep_div_indep_01': {'eq': f(x) / x * cos(f(x) / x) - (x / f(x) * sin(f(x) / x) + cos(f(x) / x)) * f(x).diff(x), 'sol': [Eq(log(x), C1 - log(f(x) * sin(f(x) / x) / x))], 'slow': True}, 'dep_div_indep_02': {'eq': x * f(x).diff(x) - f(x) - x * sin(f(x) / x), 'sol': [Eq(log(x), log(C1) + log(cos(f(x) / x) - 1) / 2 - log(cos(f(x) / x) + 1) / 2)], 'simplify_flag': False}, 'dep_div_indep_03': {'eq': x * exp(f(x) / x) - f(x) * sin(f(x) / x) + x * sin(f(x) / x) * f(x).diff(x), 'sol': [Eq(log(x), C1 + exp(-f(x) / x) * sin(f(x) / x) / 2 + exp(-f(x) / x) * cos(f(x) / x) / 2)], 'slow': True}, 'dep_div_indep_04': {'eq': f(x).diff(x) - f(x) / x + 1 / sin(f(x) / x), 'sol': [Eq(f(x), x * (-acos(C1 + log(x)) + 2 * pi)), Eq(f(x), x * acos(C1 + log(x)))], 'slow': True}, 'dep_div_indep_05': {'eq': x * exp(f(x) / x) + f(x) - x * f(x).diff(x), 'sol': [Eq(f(x), log((1 / (C1 - log(x))) ** x))], 'checkodesol_XFAIL': True}}}