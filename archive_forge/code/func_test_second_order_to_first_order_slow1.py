from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
@slow
def test_second_order_to_first_order_slow1():
    f, g = symbols('f g', cls=Function)
    x, t, x_, t_, d, a, m = symbols('x t x_ t_ d a m')
    eqs1 = [Eq(f(x).diff(x, 2), 2 / x * (x * g(x).diff(x) - g(x))), Eq(g(x).diff(x, 2), -2 / x * (x * f(x).diff(x) - f(x)))]
    sol1 = [Eq(f(x), C1 * x + 2 * C2 * x * Ci(2 * x) - C2 * sin(2 * x) - 2 * C3 * x * Si(2 * x) - C3 * cos(2 * x)), Eq(g(x), -2 * C2 * x * Si(2 * x) - C2 * cos(2 * x) - 2 * C3 * x * Ci(2 * x) + C3 * sin(2 * x) + C4 * x)]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])