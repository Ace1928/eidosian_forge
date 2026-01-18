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
def test_sysode_linear_neq_order1_type2():
    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a, b, c, d, y = symbols('x t a b c d y')
    k1, k2 = symbols('k1 k2')
    eqs1 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5), Eq(Derivative(g(x), x), -f(x) - g(x) + 7)]
    sol1 = [Eq(f(x), C1 + C2 + 6 * x ** 2 + x * (C2 + 5)), Eq(g(x), -C1 - 6 * x ** 2 - x * (C2 - 7))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5), Eq(Derivative(g(x), x), f(x) + g(x) + 7)]
    sol2 = [Eq(f(x), -C1 + C2 * exp(2 * x) - x - 3), Eq(g(x), C1 + C2 * exp(2 * x) + x - 3)]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = [Eq(Derivative(f(x), x), f(x) + 5), Eq(Derivative(g(x), x), f(x) + 7)]
    sol3 = [Eq(f(x), C1 * exp(x) - 5), Eq(g(x), C1 * exp(x) + C2 + 2 * x - 5)]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    eqs4 = [Eq(Derivative(f(x), x), f(x) + exp(x)), Eq(Derivative(g(x), x), x * exp(x) + f(x) + g(x))]
    sol4 = [Eq(f(x), C1 * exp(x) + x * exp(x)), Eq(g(x), C1 * x * exp(x) + C2 * exp(x) + x ** 2 * exp(x))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])
    eqs5 = [Eq(Derivative(f(x), x), 5 * x + f(x) + g(x)), Eq(Derivative(g(x), x), f(x) - g(x))]
    sol5 = [Eq(f(x), C1 * (1 + sqrt(2)) * exp(sqrt(2) * x) + C2 * (1 - sqrt(2)) * exp(-sqrt(2) * x) + x * Rational(-5, 2) + Rational(-5, 2)), Eq(g(x), C1 * exp(sqrt(2) * x) + C2 * exp(-sqrt(2) * x) + x * Rational(-5, 2))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])
    eqs6 = [Eq(Derivative(f(x), x), -9 * f(x) - 4 * g(x)), Eq(Derivative(g(x), x), -4 * g(x)), Eq(Derivative(h(x), x), h(x) + exp(x))]
    sol6 = [Eq(f(x), C2 * exp(-4 * x) * Rational(-4, 5) + C1 * exp(-9 * x)), Eq(g(x), C2 * exp(-4 * x)), Eq(h(x), C3 * exp(x) + x * exp(x))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0])
    eqs7 = [Eq(Derivative(f(t), t), 3 * t + f(t)), Eq(Derivative(g(t), t), g(t))]
    sol7 = [Eq(f(t), C1 * exp(t) - 3 * t - 3), Eq(g(t), C2 * exp(t))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])
    eqs8 = [Eq(Derivative(f(t), t), f(t) + 2 * g(t)), Eq(Derivative(g(t), t), -2 * f(t) + g(t) + 2 * exp(t))]
    sol8 = [Eq(f(t), C1 * exp(t) * sin(2 * t) + C2 * exp(t) * cos(2 * t) + exp(t) * sin(2 * t) ** 2 + exp(t) * cos(2 * t) ** 2), Eq(g(t), C1 * exp(t) * cos(2 * t) - C2 * exp(t) * sin(2 * t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])
    eqs9 = [Eq(Derivative(f(t), t), (c - 2 * f(t) + g(t)) / (a * b)), Eq(Derivative(g(t), t), (f(t) - 2 * g(t) + h(t)) / (a * b)), Eq(Derivative(h(t), t), (d + g(t) - 2 * h(t)) / (a * b))]
    sol9 = [Eq(f(t), -C1 * exp(-2 * t / (a * b)) + C2 * exp(-t * (sqrt(2) + 2) / (a * b)) + C3 * exp(-t * (2 - sqrt(2)) / (a * b)) + Mul(Rational(1, 4), 3 * c + d, evaluate=False)), Eq(g(t), -sqrt(2) * C2 * exp(-t * (sqrt(2) + 2) / (a * b)) + sqrt(2) * C3 * exp(-t * (2 - sqrt(2)) / (a * b)) + Mul(Rational(1, 2), c + d, evaluate=False)), Eq(h(t), C1 * exp(-2 * t / (a * b)) + C2 * exp(-t * (sqrt(2) + 2) / (a * b)) + C3 * exp(-t * (2 - sqrt(2)) / (a * b)) + Mul(Rational(1, 4), c + 3 * d, evaluate=False))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0, 0])
    eqs10 = [Eq(Derivative(f(t), t), 15 * t + f(t) - g(t) - 10), Eq(Derivative(g(t), t), -15 * t + f(t) - g(t) - 5)]
    sol10 = [Eq(f(t), C1 + C2 + 5 * t ** 3 + 5 * t ** 2 + t * (C2 - 10)), Eq(g(t), C1 + 5 * t ** 3 - 10 * t ** 2 + t * (C2 - 5))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])
    eqs11 = [Eq(Derivative(f(t), t) ** 2 - 2 * Derivative(f(t), t) + 1, 4), Eq(-y * f(t) + Derivative(g(t), t), 0)]
    sol11 = [[Eq(f(t), C1 - t), Eq(g(t), C1 * t * y + C2 * y + t ** 2 * y * Rational(-1, 2))], [Eq(f(t), C1 + 3 * t), Eq(g(t), C1 * t * y + C2 * y + t ** 2 * y * Rational(3, 2))]]
    assert dsolve(eqs11) == sol11
    for s11 in sol11:
        assert checksysodesol(eqs11, s11) == (True, [0, 0])
    n = symbols('n', positive=True)
    x0 = symbols('x_0')
    t0 = symbols('t_0')
    x_0 = symbols('x_0')
    t_0 = symbols('t_0')
    t = symbols('t')
    x = Function('x')
    y = Function('y')
    T = symbols('T')
    eqs12 = [Eq(Derivative(y(t), t), x(t)), Eq(Derivative(x(t), t), n * (y(t) + 1))]
    sol12 = [Eq(y(t), C1 * exp(sqrt(n) * t) * n ** Rational(-1, 2) - C2 * exp(-sqrt(n) * t) * n ** Rational(-1, 2) - 1), Eq(x(t), C1 * exp(sqrt(n) * t) + C2 * exp(-sqrt(n) * t))]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])
    sol12b = [Eq(y(t), (T * exp(-sqrt(n) * t_0) / 2 + exp(-sqrt(n) * t_0) / 2 + x_0 * exp(-sqrt(n) * t_0) / (2 * sqrt(n))) * exp(sqrt(n) * t) + (T * exp(sqrt(n) * t_0) / 2 + exp(sqrt(n) * t_0) / 2 - x_0 * exp(sqrt(n) * t_0) / (2 * sqrt(n))) * exp(-sqrt(n) * t) - 1), Eq(x(t), (T * sqrt(n) * exp(-sqrt(n) * t_0) / 2 + sqrt(n) * exp(-sqrt(n) * t_0) / 2 + x_0 * exp(-sqrt(n) * t_0) / 2) * exp(sqrt(n) * t) - (T * sqrt(n) * exp(sqrt(n) * t_0) / 2 + sqrt(n) * exp(sqrt(n) * t_0) / 2 - x_0 * exp(sqrt(n) * t_0) / 2) * exp(-sqrt(n) * t))]
    assert dsolve(eqs12, ics={y(t0): T, x(t0): x0}) == sol12b
    assert checksysodesol(eqs12, sol12b) == (True, [0, 0])
    eq13 = [Eq(Derivative(f(t), t), f(t) + g(t) + 9), Eq(Derivative(g(t), t), 2 * f(t) + 5 * g(t) + 23)]
    sol13 = [Eq(f(t), -C1 * (2 + sqrt(6)) * exp(t * (3 - sqrt(6))) / 2 - C2 * (2 - sqrt(6)) * exp(t * (sqrt(6) + 3)) / 2 - Rational(22, 3)), Eq(g(t), C1 * exp(t * (3 - sqrt(6))) + C2 * exp(t * (sqrt(6) + 3)) - Rational(5, 3))]
    assert dsolve(eq13) == sol13
    assert checksysodesol(eq13, sol13) == (True, [0, 0])
    eq14 = [Eq(Derivative(f(t), t), f(t) + g(t) + 81), Eq(Derivative(g(t), t), -2 * f(t) + g(t) + 23)]
    sol14 = [Eq(f(t), sqrt(2) * C1 * exp(t) * sin(sqrt(2) * t) / 2 + sqrt(2) * C2 * exp(t) * cos(sqrt(2) * t) / 2 - 58 * sin(sqrt(2) * t) ** 2 / 3 - 58 * cos(sqrt(2) * t) ** 2 / 3), Eq(g(t), C1 * exp(t) * cos(sqrt(2) * t) - C2 * exp(t) * sin(sqrt(2) * t) - 185 * sin(sqrt(2) * t) ** 2 / 3 - 185 * cos(sqrt(2) * t) ** 2 / 3)]
    assert dsolve(eq14) == sol14
    assert checksysodesol(eq14, sol14) == (True, [0, 0])
    eq15 = [Eq(Derivative(f(t), t), f(t) + 2 * g(t) + k1), Eq(Derivative(g(t), t), 3 * f(t) + 4 * g(t) + k2)]
    sol15 = [Eq(f(t), -C1 * (3 - sqrt(33)) * exp(t * (5 + sqrt(33)) / 2) / 6 - C2 * (3 + sqrt(33)) * exp(t * (5 - sqrt(33)) / 2) / 6 + 2 * k1 - k2), Eq(g(t), C1 * exp(t * (5 + sqrt(33)) / 2) + C2 * exp(t * (5 - sqrt(33)) / 2) - Mul(Rational(1, 2), 3 * k1 - k2, evaluate=False))]
    assert dsolve(eq15) == sol15
    assert checksysodesol(eq15, sol15) == (True, [0, 0])
    eq16 = [Eq(Derivative(f(t), t), k1), Eq(Derivative(g(t), t), k2)]
    sol16 = [Eq(f(t), C1 + k1 * t), Eq(g(t), C2 + k2 * t)]
    assert dsolve(eq16) == sol16
    assert checksysodesol(eq16, sol16) == (True, [0, 0])
    eq17 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), c * f(t) + k2)]
    sol17 = [Eq(f(t), C1), Eq(g(t), C2 * c + t * (C1 * c + k2))]
    assert dsolve(eq17) == sol17
    assert checksysodesol(eq17, sol17) == (True, [0, 0])
    eq18 = [Eq(Derivative(f(t), t), k1), Eq(Derivative(g(t), t), f(t) + k2)]
    sol18 = [Eq(f(t), C1 + k1 * t), Eq(g(t), C2 + k1 * t ** 2 / 2 + t * (C1 + k2))]
    assert dsolve(eq18) == sol18
    assert checksysodesol(eq18, sol18) == (True, [0, 0])
    eq19 = [Eq(Derivative(f(t), t), k1), Eq(Derivative(g(t), t), f(t) + 2 * g(t) + k2)]
    sol19 = [Eq(f(t), -2 * C1 + k1 * t), Eq(g(t), C1 + C2 * exp(2 * t) - k1 * t / 2 - Mul(Rational(1, 4), k1 + 2 * k2, evaluate=False))]
    assert dsolve(eq19) == sol19
    assert checksysodesol(eq19, sol19) == (True, [0, 0])
    eq20 = [Eq(diff(f(t), t), f(t) + k1), Eq(diff(g(t), t), k2)]
    sol20 = [Eq(f(t), C1 * exp(t) - k1), Eq(g(t), C2 + k2 * t)]
    assert dsolve(eq20) == sol20
    assert checksysodesol(eq20, sol20) == (True, [0, 0])
    eq21 = [Eq(diff(f(t), t), g(t) + k1), Eq(diff(g(t), t), 0)]
    sol21 = [Eq(f(t), C1 + t * (C2 + k1)), Eq(g(t), C2)]
    assert dsolve(eq21) == sol21
    assert checksysodesol(eq21, sol21) == (True, [0, 0])
    eq22 = [Eq(Derivative(f(t), t), f(t) + 2 * g(t) + k1), Eq(Derivative(g(t), t), k2)]
    sol22 = [Eq(f(t), -2 * C1 + C2 * exp(t) - k1 - 2 * k2 * t - 2 * k2), Eq(g(t), C1 + k2 * t)]
    assert dsolve(eq22) == sol22
    assert checksysodesol(eq22, sol22) == (True, [0, 0])
    eq23 = [Eq(Derivative(f(t), t), g(t) + k1), Eq(Derivative(g(t), t), 2 * g(t) + k2)]
    sol23 = [Eq(f(t), C1 + C2 * exp(2 * t) / 2 - k2 / 4 + t * (2 * k1 - k2) / 2), Eq(g(t), C2 * exp(2 * t) - k2 / 2)]
    assert dsolve(eq23) == sol23
    assert checksysodesol(eq23, sol23) == (True, [0, 0])
    eq24 = [Eq(Derivative(f(t), t), f(t) + k1), Eq(Derivative(g(t), t), 2 * f(t) + k2)]
    sol24 = [Eq(f(t), C1 * exp(t) / 2 - k1), Eq(g(t), C1 * exp(t) + C2 - 2 * k1 - t * (2 * k1 - k2))]
    assert dsolve(eq24) == sol24
    assert checksysodesol(eq24, sol24) == (True, [0, 0])
    eq25 = [Eq(Derivative(f(t), t), f(t) + 2 * g(t) + k1), Eq(Derivative(g(t), t), 3 * f(t) + 6 * g(t) + k2)]
    sol25 = [Eq(f(t), -2 * C1 + C2 * exp(7 * t) / 3 + 2 * t * (3 * k1 - k2) / 7 - Mul(Rational(1, 49), k1 + 2 * k2, evaluate=False)), Eq(g(t), C1 + C2 * exp(7 * t) - t * (3 * k1 - k2) / 7 - Mul(Rational(3, 49), k1 + 2 * k2, evaluate=False))]
    assert dsolve(eq25) == sol25
    assert checksysodesol(eq25, sol25) == (True, [0, 0])
    eq26 = [Eq(Derivative(f(t), t), 2 * f(t) - g(t) + k1), Eq(Derivative(g(t), t), 4 * f(t) - 2 * g(t) + 2 * k1)]
    sol26 = [Eq(f(t), C1 + 2 * C2 + t * (2 * C1 + k1)), Eq(g(t), 4 * C2 + t * (4 * C1 + 2 * k1))]
    assert dsolve(eq26) == sol26
    assert checksysodesol(eq26, sol26) == (True, [0, 0])
    eq27 = [Eq(diff(x(t), t), -1 * y(t) + 10), Eq(diff(y(t), t), 5 * x(t) - 2 * y(t) + 3)]
    sol27 = [Eq(x(t), (C1 / 5 - 2 * C2 / 5) * exp(-t) * cos(2 * t) - (2 * C1 / 5 + C2 / 5) * exp(-t) * sin(2 * t) + 17 * sin(2 * t) ** 2 / 5 + 17 * cos(2 * t) ** 2 / 5), Eq(y(t), C1 * exp(-t) * cos(2 * t) - C2 * exp(-t) * sin(2 * t) + 10 * sin(2 * t) ** 2 + 10 * cos(2 * t) ** 2)]
    assert dsolve(eq27) == sol27
    assert checksysodesol(eq27, sol27) == (True, [0, 0])