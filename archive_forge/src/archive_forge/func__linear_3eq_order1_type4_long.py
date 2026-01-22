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
def _linear_3eq_order1_type4_long():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)
    eq1 = (Eq(diff(x(t), t), (4 * f + g) * x(t) - f * y(t) - 2 * f * z(t)), Eq(diff(y(t), t), 2 * f * x(t) + (f + g) * y(t) - 2 * f * z(t)), Eq(diff(z(t), t), 5 * f * x(t) + f * y(t) + (-3 * f + g) * z(t)))
    dsolve_sol = dsolve(eq1)
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]
    x_1 = sqrt(-t ** 6 - 8 * t ** 3 * log(t) + 8 * t ** 3 - 16 * log(t) ** 2 + 32 * log(t) - 16)
    x_2 = sqrt(3)
    x_3 = 8324372644 * C1 * x_1 * x_2 + 4162186322 * C2 * x_1 * x_2 - 8324372644 * C3 * x_1 * x_2
    x_4 = 1 / (1903457163 * t ** 3 + 3825881643 * x_1 * x_2 + 7613828652 * log(t) - 7613828652)
    x_5 = exp(t ** 3 / 3 + t * x_1 * x_2 / 4 - cos(t))
    x_6 = exp(t ** 3 / 3 - t * x_1 * x_2 / 4 - cos(t))
    x_7 = exp(t ** 4 / 2 + t ** 3 / 3 + 2 * t * log(t) - 2 * t - cos(t))
    x_8 = 91238 * C1 * x_1 * x_2 + 91238 * C2 * x_1 * x_2 - 91238 * C3 * x_1 * x_2
    x_9 = 1 / (66049 * t ** 3 - 50629 * x_1 * x_2 + 264196 * log(t) - 264196)
    x_10 = 50629 * C1 / 25189 + 37909 * C2 / 25189 - 50629 * C3 / 25189 - x_3 * x_4
    x_11 = -50629 * C1 / 25189 - 12720 * C2 / 25189 + 50629 * C3 / 25189 + x_3 * x_4
    sol = [Eq(x(t), x_10 * x_5 + x_11 * x_6 + x_7 * (C1 - C2)), Eq(y(t), x_10 * x_5 + x_11 * x_6), Eq(z(t), x_5 * (-424 * C1 / 257 - 167 * C2 / 257 + 424 * C3 / 257 - x_8 * x_9) + x_6 * (167 * C1 / 257 + 424 * C2 / 257 - 167 * C3 / 257 + x_8 * x_9) + x_7 * (C1 - C2))]
    assert dsolve_sol1 == sol
    assert checksysodesol(eq1, dsolve_sol1) == (True, [0, 0, 0])