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
def test_nth_order_linear_euler_eq_homogeneous():
    x, t, a, b, c = symbols('x t a b c')
    y = Function('y')
    our_hint = 'nth_linear_euler_eq_homogeneous'
    eq = diff(f(t), t, 4) * t ** 4 - 13 * diff(f(t), t, 2) * t ** 2 + 36 * f(t)
    assert our_hint in classify_ode(eq)
    eq = a * y(t) + b * t * diff(y(t), t) + c * t ** 2 * diff(y(t), t, 2)
    assert our_hint in classify_ode(eq)
    _ode_solver_test(_get_examples_ode_sol_euler_homogeneous)