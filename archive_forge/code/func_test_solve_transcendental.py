from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
@slow
def test_solve_transcendental():
    from sympy.abc import a, b
    assert solve(exp(x) - 3, x) == [log(3)]
    assert set(solve((a * x + b) * (exp(x) - 3), x)) == {-b / a, log(3)}
    assert solve(cos(x) - y, x) == [-acos(y) + 2 * pi, acos(y)]
    assert solve(2 * cos(x) - y, x) == [-acos(y / 2) + 2 * pi, acos(y / 2)]
    assert solve(Eq(cos(x), sin(x)), x) == [pi / 4]
    assert set(solve(exp(x) + exp(-x) - y, x)) in [{log(y / 2 - sqrt(y ** 2 - 4) / 2), log(y / 2 + sqrt(y ** 2 - 4) / 2)}, {log(y - sqrt(y ** 2 - 4)) - log(2), log(y + sqrt(y ** 2 - 4)) - log(2)}, {log(y / 2 - sqrt((y - 2) * (y + 2)) / 2), log(y / 2 + sqrt((y - 2) * (y + 2)) / 2)}]
    assert solve(exp(x) - 3, x) == [log(3)]
    assert solve(Eq(exp(x), 3), x) == [log(3)]
    assert solve(log(x) - 3, x) == [exp(3)]
    assert solve(sqrt(3 * x) - 4, x) == [Rational(16, 3)]
    assert solve(3 ** (x + 2), x) == []
    assert solve(3 ** (2 - x), x) == []
    assert solve(x + 2 ** x, x) == [-LambertW(log(2)) / log(2)]
    assert solve(2 * x + 5 + log(3 * x - 2), x) == [Rational(2, 3) + LambertW(2 * exp(Rational(-19, 3)) / 3) / 2]
    assert solve(3 * x + log(4 * x), x) == [LambertW(Rational(3, 4)) / 3]
    assert set(solve((2 * x + 8) * (8 + exp(x)), x)) == {S(-4), log(8) + pi * I}
    eq = 2 * exp(3 * x + 4) - 3
    ans = solve(eq, x)
    assert len(ans) == 3 and all((eq.subs(x, a).n(chop=True) == 0 for a in ans))
    assert solve(2 * log(3 * x + 4) - 3, x) == [(exp(Rational(3, 2)) - 4) / 3]
    assert solve(exp(x) + 1, x) == [pi * I]
    eq = 2 * (3 * x + 4) ** 5 - 6 * 7 ** (3 * x + 9)
    result = solve(eq, x)
    x0 = -log(2401)
    x1 = 3 ** Rational(1, 5)
    x2 = log(7 ** (7 * x1 / 20))
    x3 = sqrt(2)
    x4 = sqrt(5)
    x5 = x3 * sqrt(x4 - 5)
    x6 = x4 + 1
    x7 = 1 / (3 * log(7))
    x8 = -x4
    x9 = x3 * sqrt(x8 - 5)
    x10 = x8 + 1
    ans = [x7 * (x0 - 5 * LambertW(x2 * (-x5 + x6))), x7 * (x0 - 5 * LambertW(x2 * (x5 + x6))), x7 * (x0 - 5 * LambertW(x2 * (x10 - x9))), x7 * (x0 - 5 * LambertW(x2 * (x10 + x9))), x7 * (x0 - 5 * LambertW(-log(7 ** (7 * x1 / 5))))]
    assert result == ans, result
    assert solve(eq.expand(), x) == result
    assert solve(z * cos(x) - y, x) == [-acos(y / z) + 2 * pi, acos(y / z)]
    assert solve(z * cos(2 * x) - y, x) == [-acos(y / z) / 2 + pi, acos(y / z) / 2]
    assert solve(z * cos(sin(x)) - y, x) == [pi - asin(acos(y / z)), asin(acos(y / z) - 2 * pi) + pi, -asin(acos(y / z) - 2 * pi), asin(acos(y / z))]
    assert solve(z * cos(x), x) == [pi / 2, pi * Rational(3, 2)]
    assert solve(y - b * x / (a + x), x) in [[-a * y / (y - b)], [a * y / (b - y)]]
    assert solve(y - b * exp(a / x), x) == [a / log(y / b)]
    assert solve(y - b / (1 + a * x), x) in [[(b - y) / (a * y)], [-((y - b) / (a * y))]]
    assert solve(y - a * x ** b, x) == [(y / a) ** (1 / b)]
    assert solve(z ** x - y, x) == [log(y) / log(z)]
    assert solve(2 ** x - 10, x) == [1 + log(5) / log(2)]
    assert solve(x * y) == [{x: 0}, {y: 0}]
    assert solve([x * y]) == [{x: 0}, {y: 0}]
    assert solve(x ** y - 1) == [{x: 1}, {y: 0}]
    assert solve([x ** y - 1]) == [{x: 1}, {y: 0}]
    assert solve(x * y * (x ** 2 - y ** 2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve([x * y * (x ** 2 - y ** 2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve(exp(log(5) * x) - 2 ** x, x) == [0]
    assert solve(exp(log(5) * x) - exp(log(2) * x), x) == [0]
    f = Function('f')
    assert solve(y * f(log(5) * x) - y * f(log(2) * x), x) == [0]
    assert solve(f(x) - f(0), x) == [0]
    assert solve(f(x) - f(2 - x), x) == [1]
    raises(NotImplementedError, lambda: solve(f(x, y) - f(1, 2), x))
    raises(NotImplementedError, lambda: solve(f(x, y) - f(2 - x, 2), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1 - x), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1), x))
    raises(NotImplementedError, lambda: solve(sinh(x) * sinh(sinh(x)) + cosh(x) * cosh(sinh(x)) - 3))
    raises(NotImplementedError, lambda: solve((x + 2) ** y * x - 3, x))
    assert solve(sin(sqrt(x))) == [0, pi ** 2]
    a, b = symbols('a, b', real=True, negative=False)
    assert str(solve(Eq(a, 0.5 - cos(pi * b) / 2), b)) == '[2.0 - 0.318309886183791*acos(1.0 - 2.0*a), 0.318309886183791*acos(1.0 - 2.0*a)]'
    assert solve(y ** (1 / x) - z, x) == [log(y) / log(z)]