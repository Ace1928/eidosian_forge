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
def test_nonlinear_2eq_order1():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    eq1 = (Eq(diff(x(t), t), x(t) * y(t) ** 3), Eq(diff(y(t), t), y(t) ** 5))
    sol1 = [Eq(x(t), C1 * exp((-1 / (4 * C2 + 4 * t)) ** Rational(-1, 4))), Eq(y(t), -(-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), C1 * exp(-1 / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), C1 * exp(-I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), -I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), C1 * exp(I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))]
    assert dsolve(eq1) == sol1
    assert checksysodesol(eq1, sol1) == (True, [0, 0])
    eq2 = (Eq(diff(x(t), t), exp(3 * x(t)) * y(t) ** 3), Eq(diff(y(t), t), y(t) ** 5))
    sol2 = [Eq(x(t), -log(C1 - 3 / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)) / 3), Eq(y(t), -(-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), -log(C1 + 3 / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)) / 3), Eq(y(t), (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), -log(C1 + 3 * I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)) / 3), Eq(y(t), -I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), -log(C1 - 3 * I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)) / 3), Eq(y(t), I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))]
    assert dsolve(eq2) == sol2
    assert checksysodesol(eq2, sol2) == (True, [0, 0])
    eq3 = (Eq(diff(x(t), t), y(t) * x(t)), Eq(diff(y(t), t), x(t) ** 3))
    tt = Rational(2, 3)
    sol3 = [Eq(x(t), 6 ** tt / (6 * (-sinh(sqrt(C1) * (C2 + t) / 2) / sqrt(C1)) ** tt)), Eq(y(t), sqrt(C1 + C1 / sinh(sqrt(C1) * (C2 + t) / 2) ** 2) / 3)]
    assert dsolve(eq3) == sol3
    eq4 = (Eq(diff(x(t), t), x(t) * y(t) * sin(t) ** 2), Eq(diff(y(t), t), y(t) ** 2 * sin(t) ** 2))
    sol4 = {Eq(x(t), -2 * exp(C1) / (C2 * exp(C1) + t - sin(2 * t) / 2)), Eq(y(t), -2 / (C1 + t - sin(2 * t) / 2))}
    assert dsolve(eq4) == sol4
    eq5 = (Eq(x(t), t * diff(x(t), t) + diff(x(t), t) * diff(y(t), t)), Eq(y(t), t * diff(y(t), t) + diff(y(t), t) ** 2))
    sol5 = {Eq(x(t), C1 * C2 + C1 * t), Eq(y(t), C2 ** 2 + C2 * t)}
    assert dsolve(eq5) == sol5
    assert checksysodesol(eq5, sol5) == (True, [0, 0])
    eq6 = (Eq(diff(x(t), t), x(t) ** 2 * y(t) ** 3), Eq(diff(y(t), t), y(t) ** 5))
    sol6 = [Eq(x(t), 1 / (C1 - 1 / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), -(-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), 1 / (C1 + (-1 / (4 * C2 + 4 * t)) ** Rational(-1, 4))), Eq(y(t), (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), 1 / (C1 + I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), -I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4)), Eq(x(t), 1 / (C1 - I / (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))), Eq(y(t), I * (-1 / (4 * C2 + 4 * t)) ** Rational(1, 4))]
    assert dsolve(eq6) == sol6
    assert checksysodesol(eq6, sol6) == (True, [0, 0])