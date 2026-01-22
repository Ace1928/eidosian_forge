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
def _neq_order1_type4_slow3():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')
    eqs = [Eq(Derivative(f(x), x), x * f(x) + g(x) + sin(x)), Eq(Derivative(g(x), x), x ** 2 + x * g(x) - f(x))]
    sol = [Eq(f(x), (C1 / 2 - I * C2 / 2 - I * Integral(x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + I * exp(-x ** 2 / 2 - I * x) * sin(x) / 2 - I * exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2 + Integral(-I * x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + I * x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + exp(-x ** 2 / 2 - I * x) * sin(x) / 2 + exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2) * exp(x ** 2 / 2 + I * x) + (C1 / 2 + I * C2 / 2 + I * Integral(x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + I * exp(-x ** 2 / 2 - I * x) * sin(x) / 2 - I * exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2 + Integral(-I * x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + I * x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + exp(-x ** 2 / 2 - I * x) * sin(x) / 2 + exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2) * exp(x ** 2 / 2 - I * x)), Eq(g(x), (-I * C1 / 2 + C2 / 2 + Integral(x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + I * exp(-x ** 2 / 2 - I * x) * sin(x) / 2 - I * exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2 - I * Integral(-I * x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + I * x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + exp(-x ** 2 / 2 - I * x) * sin(x) / 2 + exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2) * exp(x ** 2 / 2 - I * x) + (I * C1 / 2 + C2 / 2 + Integral(x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + I * exp(-x ** 2 / 2 - I * x) * sin(x) / 2 - I * exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2 + I * Integral(-I * x ** 2 * exp(-x ** 2 / 2 - I * x) / 2 + I * x ** 2 * exp(-x ** 2 / 2 + I * x) / 2 + exp(-x ** 2 / 2 - I * x) * sin(x) / 2 + exp(-x ** 2 / 2 + I * x) * sin(x) / 2, x) / 2) * exp(x ** 2 / 2 + I * x))]
    return (eqs, sol)