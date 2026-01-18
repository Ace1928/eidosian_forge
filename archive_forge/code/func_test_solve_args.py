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
def test_solve_args():
    ans = {x: -3, y: 1}
    eqs = (x + 5 * y - 2, -3 * x + 6 * y - 15)
    assert all((solve(container(eqs), x, y) == ans for container in (tuple, list, set, frozenset)))
    assert solve(Tuple(*eqs), x, y) == ans
    assert set(solve(x ** 2 - 4)) == {S(2), -S(2)}
    assert solve([x + y - 3, x - y - 5]) == {x: 4, y: -1}
    assert solve(x - exp(x), x, implicit=True) == [exp(x)]
    assert solve(42) == solve(42, x) == []
    assert solve([1, 2]) == []
    assert solve([sqrt(2)], [x]) == []
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    assert solve(x, x, exclude=[y, y]) == [0]
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    assert solve(x, x, exclude=[y, y]) == [0]
    assert solve(y - 3, {y}) == [3]
    assert solve(y - 3, {x, y}) == [{y: 3}]
    assert solve(x + y - 3, [x, y]) == [(3 - y, y)]
    assert solve(x + y - 3, [x, y], dict=True) == [{x: 3 - y}]
    assert solve(x + y - 3) == [{x: 3 - y}]
    assert solve(a + b * x - 2, [a, b]) == {a: 2, b: 0}
    assert solve((a + b) * x + b - c, [a, b]) == {a: -c, b: c}
    eq = a * x ** 2 + b * x + c - ((x - h) ** 2 + 4 * p * k) / 4 / p
    sol = solve(eq, [h, p, k], exclude=[a, b, c])
    assert sol == {h: -b / (2 * a), k: (4 * a * c - b ** 2) / (4 * a), p: 1 / (4 * a)}
    assert solve(eq, [h, p, k], dict=True) == [sol]
    assert solve(eq, [h, p, k], set=True) == ([h, p, k], {(-b / (2 * a), 1 / (4 * a), (4 * a * c - b ** 2) / (4 * a))})
    assert solve(eq, [h, p, k], exclude=[a, b, c], simplify=False) == {h: -b / (2 * a), k: (4 * a * c - b ** 2) / (4 * a), p: 1 / (4 * a)}
    args = ((a + b) * x - b ** 2 + 2, a, b)
    assert solve(*args) == [((b ** 2 - b * x - 2) / x, b)]
    assert solve(a * x + b ** 2 / (x + 4) - 3 * x - 4 / x, a, b, dict=True) == [{a: (-b ** 2 * x + 3 * x ** 3 + 12 * x ** 2 + 4 * x + 16) / (x ** 2 * (x + 4))}]
    assert solve(1 / (1 / x - y + exp(y))) == []
    raises(NotImplementedError, lambda: solve(exp(x) + sin(x) + exp(y) + sin(y)))
    assert solve([y, exp(x) + x]) == [{x: -LambertW(1), y: 0}]
    assert solve((exp(x) - x, exp(y) - y)) == [{x: -LambertW(-1), y: -LambertW(-1)}]
    assert solve([y, exp(x) + x], x, y) == [(-LambertW(1), 0)]
    assert solve(x ** 2 - pi, pi) == [x ** 2]
    assert solve([], [x]) == []
    assert solve((x ** 2 - 4, y - 2), x, y) == [(-2, 2), (2, 2)]
    assert solve((x ** 2 - 4, y - 2), y, x) == [(2, -2), (2, 2)]
    assert solve((x ** 2 - 4 + z, y - 2 - z), a, z, y, x, set=True) == ([a, z, y, x], {(a, z, z + 2, -sqrt(4 - z)), (a, z, z + 2, sqrt(4 - z))})
    assert solve([(x + y) ** 2 - 4, x + y - 2]) == [{x: -y + 2}]
    assert solve((x + y - 2, 2 * x + 2 * y - 4)) == {x: -y + 2}
    assert solve(Eq(x ** 2, 0.0)) == [0.0]
    assert solve([True, Eq(x, 0)], [x], dict=True) == [{x: 0}]
    assert solve([Eq(x, x), Eq(x, 0), Eq(x, x + 1)], [x], dict=True) == []
    assert not solve([Eq(x, x + 1), x < 2], x)
    assert solve([Eq(x, 0), x + 1 < 2]) == Eq(x, 0)
    assert solve([Eq(x, x), Eq(x, x + 1)], x) == []
    assert solve(True, x) == []
    assert solve([x - 1, False], [x], set=True) == ([], set())
    assert solve([-y * (x + y - 1) / 2, (y - 1) / x / y + 1 / y], set=True, check=False) == ([x, y], {(1 - y, y), (x, 0)})
    assert list(solve((y - 1, x - sqrt(3) * z)).keys()) == [x, y]
    assert solve([x - 1, x], (y, x), set=True) == ([y, x], set())
    assert solve([x - 1, x], {y, x}, set=True) == ([x, y], set())