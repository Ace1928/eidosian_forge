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
def _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved():
    return {'hint': '2nd_nonlinear_autonomous_conserved', 'func': f(x), 'examples': {'2nd_nonlinear_autonomous_conserved_01': {'eq': f(x).diff(x, 2) + exp(f(x)) + log(f(x)), 'sol': [Eq(Integral(1 / sqrt(C1 - 2 * _u * log(_u) + 2 * _u - 2 * exp(_u)), (_u, f(x))), C2 + x), Eq(Integral(1 / sqrt(C1 - 2 * _u * log(_u) + 2 * _u - 2 * exp(_u)), (_u, f(x))), C2 - x)], 'simplify_flag': False}, '2nd_nonlinear_autonomous_conserved_02': {'eq': f(x).diff(x, 2) + cbrt(f(x)) + 1 / f(x), 'sol': [Eq(sqrt(2) * Integral(1 / sqrt(2 * C1 - 3 * _u ** Rational(4, 3) - 4 * log(_u)), (_u, f(x))), C2 + x), Eq(sqrt(2) * Integral(1 / sqrt(2 * C1 - 3 * _u ** Rational(4, 3) - 4 * log(_u)), (_u, f(x))), C2 - x)], 'simplify_flag': False}, '2nd_nonlinear_autonomous_conserved_03': {'eq': f(x).diff(x, 2) + sin(f(x)), 'sol': [Eq(Integral(1 / sqrt(C1 + 2 * cos(_u)), (_u, f(x))), C2 + x), Eq(Integral(1 / sqrt(C1 + 2 * cos(_u)), (_u, f(x))), C2 - x)], 'simplify_flag': False}, '2nd_nonlinear_autonomous_conserved_04': {'eq': f(x).diff(x, 2) + cosh(f(x)), 'sol': [Eq(Integral(1 / sqrt(C1 - 2 * sinh(_u)), (_u, f(x))), C2 + x), Eq(Integral(1 / sqrt(C1 - 2 * sinh(_u)), (_u, f(x))), C2 - x)], 'simplify_flag': False}, '2nd_nonlinear_autonomous_conserved_05': {'eq': f(x).diff(x, 2) + asin(f(x)), 'sol': [Eq(Integral(1 / sqrt(C1 - 2 * _u * asin(_u) - 2 * sqrt(1 - _u ** 2)), (_u, f(x))), C2 + x), Eq(Integral(1 / sqrt(C1 - 2 * _u * asin(_u) - 2 * sqrt(1 - _u ** 2)), (_u, f(x))), C2 - x)], 'simplify_flag': False, 'XFAIL': ['2nd_nonlinear_autonomous_conserved_Integral']}}}