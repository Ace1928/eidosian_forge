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
def _get_examples_ode_sol_almost_linear():
    from sympy.functions.special.error_functions import Ei
    A = Symbol('A', positive=True)
    f = Function('f')
    d = f(x).diff(x)
    return {'hint': 'almost_linear', 'func': f(x), 'examples': {'almost_lin_01': {'eq': x ** 2 * f(x) ** 2 * d + f(x) ** 3 + 1, 'sol': [Eq(f(x), (C1 * exp(3 / x) - 1) ** Rational(1, 3)), Eq(f(x), (-1 - sqrt(3) * I) * (C1 * exp(3 / x) - 1) ** Rational(1, 3) / 2), Eq(f(x), (-1 + sqrt(3) * I) * (C1 * exp(3 / x) - 1) ** Rational(1, 3) / 2)]}, 'almost_lin_02': {'eq': x * f(x) * d + 2 * x * f(x) ** 2 + 1, 'sol': [Eq(f(x), -sqrt((C1 - 2 * Ei(4 * x)) * exp(-4 * x))), Eq(f(x), sqrt((C1 - 2 * Ei(4 * x)) * exp(-4 * x)))]}, 'almost_lin_03': {'eq': x * d + x * f(x) + 1, 'sol': [Eq(f(x), (C1 - Ei(x)) * exp(-x))]}, 'almost_lin_04': {'eq': x * exp(f(x)) * d + exp(f(x)) + 3 * x, 'sol': [Eq(f(x), log(C1 / x - x * Rational(3, 2)))]}, 'almost_lin_05': {'eq': x + A * (x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2, 'sol': [Eq(f(x), (C1 + Piecewise((x, Eq(A + 1, 0)), ((-A * x + A - x - 1) * exp(x) / (A + 1), True))) * exp(-x))]}}}