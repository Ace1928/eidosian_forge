from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
def test_solve_riccati():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.

    Some examples have been taken from the paper - "Statistical Investigation of
    First-Order Algebraic ODEs and their Rational General Solutions" by
    Georg Grasegger, N. Thieu Vo, Franz Winkler

    https://www3.risc.jku.at/publications/download/risc_5197/RISCReport15-19.pdf
    """
    C0 = Dummy('C0')
    tests = [(Eq(f(x).diff(x) + f(x) ** 2 - 2, 0), [Eq(f(x), sqrt(2)), Eq(f(x), -sqrt(2))]), (f(x) ** 2 + f(x).diff(x) + 4 * f(x) / x + 2 / x ** 2, [Eq(f(x), (-2 * C0 - x) / (C0 * x + x ** 2))]), (2 * x ** 2 * f(x).diff(x) - x * (4 * f(x) + f(x).diff(x) - 4) + (f(x) - 1) * f(x), [Eq(f(x), (C0 + 2 * x ** 2) / (C0 + x))]), (Eq(f(x).diff(x), -f(x) ** 2 - 2 / (x ** 3 - x ** 2)), [Eq(f(x), 1 / (x ** 2 - x))]), (x ** 2 - (2 * x + 1 / x) * f(x) + f(x) ** 2 + f(x).diff(x), [Eq(f(x), (C0 * x + x ** 3 + 2 * x) / (C0 + x ** 2)), Eq(f(x), x)]), (x ** 4 * f(x).diff(x) + x ** 2 - x * (2 * f(x) ** 2 + f(x).diff(x)) + f(x), [Eq(f(x), (C0 * x ** 2 + x) / (C0 + x ** 2)), Eq(f(x), x ** 2)]), (-f(x) ** 2 + f(x).diff(x) + (15 * x ** 2 - 20 * x + 7) / ((x - 1) ** 2 * (2 * x - 1) ** 2), [Eq(f(x), (9 * C0 * x - 6 * C0 - 15 * x ** 5 + 60 * x ** 4 - 94 * x ** 3 + 72 * x ** 2 - 30 * x + 6) / (6 * C0 * x ** 2 - 9 * C0 * x + 3 * C0 + 6 * x ** 6 - 29 * x ** 5 + 57 * x ** 4 - 58 * x ** 3 + 30 * x ** 2 - 6 * x)), Eq(f(x), (3 * x - 2) / (2 * x ** 2 - 3 * x + 1))]), (f(x) ** 2 + f(x).diff(x) - (4 * x ** 6 - 8 * x ** 5 + 12 * x ** 4 + 4 * x ** 3 + 7 * x ** 2 - 20 * x + 4) / (4 * x ** 4), [Eq(f(x), (2 * x ** 5 - 2 * x ** 4 - x ** 3 + 4 * x ** 2 + 3 * x - 2) / (2 * x ** 4 - 2 * x ** 2))]), (Eq(f(x).diff(x), (-x ** 6 + 15 * x ** 4 - 40 * x ** 3 + 45 * x ** 2 - 24 * x + 4) / (x ** 12 - 12 * x ** 11 + 66 * x ** 10 - 220 * x ** 9 + 495 * x ** 8 - 792 * x ** 7 + 924 * x ** 6 - 792 * x ** 5 + 495 * x ** 4 - 220 * x ** 3 + 66 * x ** 2 - 12 * x + 1) + f(x) ** 2 + f(x)), [Eq(f(x), 1 / (x ** 6 - 6 * x ** 5 + 15 * x ** 4 - 20 * x ** 3 + 15 * x ** 2 - 6 * x + 1))]), (Eq(f(x).diff(x), x * f(x) + 2 * x + (3 * x - 2) * f(x) ** 2 / (4 * x + 2) + (8 * x ** 2 - 7 * x + 26) / (16 * x ** 3 - 24 * x ** 2 + 8) - S(3) / 2), [Eq(f(x), (1 - 4 * x) / (2 * x - 2))]), (Eq(f(x).diff(x), (-12 * x ** 2 - 48 * x - 15) / (24 * x ** 3 - 40 * x ** 2 + 8 * x + 8) + 3 * f(x) ** 2 / (6 * x + 2)), [Eq(f(x), (2 * x + 1) / (2 * x - 2))]), (f(x).diff(x) + (3 * x ** 2 + 1) * f(x) ** 2 / x + (6 * x ** 2 - x + 3) * f(x) / (x * (x - 1)) + (3 * x ** 2 - 2 * x + 2) / (x * (x - 1) ** 2), [Eq(f(x), (-C0 - x ** 3 + x ** 2 - 2 * x) / (C0 * x - C0 + x ** 4 - x ** 3 + x ** 2 - x)), Eq(f(x), -1 / (x - 1))]), (f(x).diff(x) - 2 * I * (f(x) ** 2 + 1) / x, [Eq(f(x), (-I * C0 + I * x ** 4) / (C0 + x ** 4)), Eq(f(x), -I)]), (Eq(f(x).diff(x), x * f(x) / (S(3) / 2 - 2 * x) + (x / 2 - S(1) / 3) * f(x) ** 2 / (2 * x / 3 - S(1) / 2) - S(5) / 4 + (281 * x ** 2 - 1260 * x + 756) / (16 * x ** 3 - 12 * x ** 2)), [Eq(f(x), (9 - x) / x), Eq(f(x), (40 * x ** 14 + 28 * x ** 13 + 420 * x ** 12 + 2940 * x ** 11 + 18480 * x ** 10 + 103950 * x ** 9 + 519750 * x ** 8 + 2286900 * x ** 7 + 8731800 * x ** 6 + 28378350 * x ** 5 + 76403250 * x ** 4 + 163721250 * x ** 3 + 261954000 * x ** 2 + 278326125 * x + 147349125) / (24 * x ** 14 + 140 * x ** 13 + 840 * x ** 12 + 4620 * x ** 11 + 23100 * x ** 10 + 103950 * x ** 9 + 415800 * x ** 8 + 1455300 * x ** 7 + 4365900 * x ** 6 + 10914750 * x ** 5 + 21829500 * x ** 4 + 32744250 * x ** 3 + 32744250 * x ** 2 + 16372125 * x))]), (Eq(f(x).diff(x), 18 * x ** 3 + 18 * x ** 2 + (-x / 2 - S(1) / 2) * f(x) ** 2 + 6), [Eq(f(x), 6 * x)]), (Eq(f(x).diff(x), -3 * x ** 3 / 4 + 15 * x / 2 + (x / 3 - S(4) / 3) * f(x) ** 2 + 9 + (1 - x) * f(x) / x + 3 / x), [Eq(f(x), -3 * x / 2 - 3)])]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)