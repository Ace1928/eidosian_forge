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
def test_sysode_linear_neq_order1_type1():
    f, g, x, y, h = symbols('f g x y h', cls=Function)
    a, b, c, t = symbols('a b c t')
    eqs1 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), y(t))]
    sol1 = [Eq(x(t), C1 * exp(t)), Eq(y(t), C2 * exp(t))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(Derivative(x(t), t), 2 * x(t)), Eq(Derivative(y(t), t), 3 * y(t))]
    sol2 = [Eq(x(t), C1 * exp(2 * t)), Eq(y(t), C2 * exp(3 * t))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = [Eq(Derivative(x(t), t), a * x(t)), Eq(Derivative(y(t), t), a * y(t))]
    sol3 = [Eq(x(t), C1 * exp(a * t)), Eq(y(t), C2 * exp(a * t))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    eqs4 = [Eq(Derivative(x(t), t), a * x(t)), Eq(Derivative(y(t), t), b * y(t))]
    sol4 = [Eq(x(t), C1 * exp(a * t)), Eq(y(t), C2 * exp(b * t))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])
    eqs5 = [Eq(Derivative(x(t), t), -y(t)), Eq(Derivative(y(t), t), x(t))]
    sol5 = [Eq(x(t), -C1 * sin(t) - C2 * cos(t)), Eq(y(t), C1 * cos(t) - C2 * sin(t))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])
    eqs6 = [Eq(Derivative(x(t), t), -2 * y(t)), Eq(Derivative(y(t), t), 2 * x(t))]
    sol6 = [Eq(x(t), -C1 * sin(2 * t) - C2 * cos(2 * t)), Eq(y(t), C1 * cos(2 * t) - C2 * sin(2 * t))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])
    eqs7 = [Eq(Derivative(x(t), t), I * y(t)), Eq(Derivative(y(t), t), I * x(t))]
    sol7 = [Eq(x(t), -C1 * exp(-I * t) + C2 * exp(I * t)), Eq(y(t), C1 * exp(-I * t) + C2 * exp(I * t))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])
    eqs8 = [Eq(Derivative(x(t), t), -a * y(t)), Eq(Derivative(y(t), t), a * x(t))]
    sol8 = [Eq(x(t), -I * C1 * exp(-I * a * t) + I * C2 * exp(I * a * t)), Eq(y(t), C1 * exp(-I * a * t) + C2 * exp(I * a * t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])
    eqs9 = [Eq(Derivative(x(t), t), x(t) + y(t)), Eq(Derivative(y(t), t), x(t) - y(t))]
    sol9 = [Eq(x(t), C1 * (1 - sqrt(2)) * exp(-sqrt(2) * t) + C2 * (1 + sqrt(2)) * exp(sqrt(2) * t)), Eq(y(t), C1 * exp(-sqrt(2) * t) + C2 * exp(sqrt(2) * t))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])
    eqs10 = [Eq(Derivative(x(t), t), x(t) + y(t)), Eq(Derivative(y(t), t), x(t) + y(t))]
    sol10 = [Eq(x(t), -C1 + C2 * exp(2 * t)), Eq(y(t), C1 + C2 * exp(2 * t))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])
    eqs11 = [Eq(Derivative(x(t), t), 2 * x(t) + y(t)), Eq(Derivative(y(t), t), -x(t) + 2 * y(t))]
    sol11 = [Eq(x(t), C1 * exp(2 * t) * sin(t) + C2 * exp(2 * t) * cos(t)), Eq(y(t), C1 * exp(2 * t) * cos(t) - C2 * exp(2 * t) * sin(t))]
    assert dsolve(eqs11) == sol11
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])
    eqs12 = [Eq(Derivative(x(t), t), x(t) + 2 * y(t)), Eq(Derivative(y(t), t), 2 * x(t) + y(t))]
    sol12 = [Eq(x(t), -C1 * exp(-t) + C2 * exp(3 * t)), Eq(y(t), C1 * exp(-t) + C2 * exp(3 * t))]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])
    eqs13 = [Eq(Derivative(x(t), t), 4 * x(t) + y(t)), Eq(Derivative(y(t), t), -x(t) + 2 * y(t))]
    sol13 = [Eq(x(t), C2 * t * exp(3 * t) + (C1 + C2) * exp(3 * t)), Eq(y(t), -C1 * exp(3 * t) - C2 * t * exp(3 * t))]
    assert dsolve(eqs13) == sol13
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])
    eqs14 = [Eq(Derivative(x(t), t), a * y(t)), Eq(Derivative(y(t), t), a * x(t))]
    sol14 = [Eq(x(t), -C1 * exp(-a * t) + C2 * exp(a * t)), Eq(y(t), C1 * exp(-a * t) + C2 * exp(a * t))]
    assert dsolve(eqs14) == sol14
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])
    eqs15 = [Eq(Derivative(x(t), t), a * y(t)), Eq(Derivative(y(t), t), b * x(t))]
    sol15 = [Eq(x(t), -C1 * a * exp(-t * sqrt(a * b)) / sqrt(a * b) + C2 * a * exp(t * sqrt(a * b)) / sqrt(a * b)), Eq(y(t), C1 * exp(-t * sqrt(a * b)) + C2 * exp(t * sqrt(a * b)))]
    assert dsolve(eqs15) == sol15
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])
    eqs16 = [Eq(Derivative(x(t), t), a * x(t) + b * y(t)), Eq(Derivative(y(t), t), c * x(t))]
    sol16 = [Eq(x(t), -2 * C1 * b * exp(t * (a + sqrt(a ** 2 + 4 * b * c)) / 2) / (a - sqrt(a ** 2 + 4 * b * c)) - 2 * C2 * b * exp(t * (a - sqrt(a ** 2 + 4 * b * c)) / 2) / (a + sqrt(a ** 2 + 4 * b * c))), Eq(y(t), C1 * exp(t * (a + sqrt(a ** 2 + 4 * b * c)) / 2) + C2 * exp(t * (a - sqrt(a ** 2 + 4 * b * c)) / 2))]
    assert dsolve(eqs16) == sol16
    assert checksysodesol(eqs16, sol16) == (True, [0, 0])
    eqs17 = [Eq(Derivative(x(t), t), a * y(t) + x(t)), Eq(Derivative(y(t), t), a * x(t) - y(t))]
    sol17 = [Eq(x(t), C1 * a * exp(t * sqrt(a ** 2 + 1)) / (sqrt(a ** 2 + 1) - 1) - C2 * a * exp(-t * sqrt(a ** 2 + 1)) / (sqrt(a ** 2 + 1) + 1)), Eq(y(t), C1 * exp(t * sqrt(a ** 2 + 1)) + C2 * exp(-t * sqrt(a ** 2 + 1)))]
    assert dsolve(eqs17) == sol17
    assert checksysodesol(eqs17, sol17) == (True, [0, 0])
    eqs18 = [Eq(Derivative(x(t), t), 0), Eq(Derivative(y(t), t), 0)]
    sol18 = [Eq(x(t), C1), Eq(y(t), C2)]
    assert dsolve(eqs18) == sol18
    assert checksysodesol(eqs18, sol18) == (True, [0, 0])
    eqs19 = [Eq(Derivative(x(t), t), 2 * x(t) - y(t)), Eq(Derivative(y(t), t), x(t))]
    sol19 = [Eq(x(t), C2 * t * exp(t) + (C1 + C2) * exp(t)), Eq(y(t), C1 * exp(t) + C2 * t * exp(t))]
    assert dsolve(eqs19) == sol19
    assert checksysodesol(eqs19, sol19) == (True, [0, 0])
    eqs20 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), x(t) + y(t))]
    sol20 = [Eq(x(t), C1 * exp(t)), Eq(y(t), C1 * t * exp(t) + C2 * exp(t))]
    assert dsolve(eqs20) == sol20
    assert checksysodesol(eqs20, sol20) == (True, [0, 0])
    eqs21 = [Eq(Derivative(x(t), t), 3 * x(t)), Eq(Derivative(y(t), t), x(t) + y(t))]
    sol21 = [Eq(x(t), 2 * C1 * exp(3 * t)), Eq(y(t), C1 * exp(3 * t) + C2 * exp(t))]
    assert dsolve(eqs21) == sol21
    assert checksysodesol(eqs21, sol21) == (True, [0, 0])
    eqs22 = [Eq(Derivative(x(t), t), 3 * x(t)), Eq(Derivative(y(t), t), y(t))]
    sol22 = [Eq(x(t), C1 * exp(3 * t)), Eq(y(t), C2 * exp(t))]
    assert dsolve(eqs22) == sol22
    assert checksysodesol(eqs22, sol22) == (True, [0, 0])