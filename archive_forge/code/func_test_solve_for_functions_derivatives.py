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
def test_solve_for_functions_derivatives():
    t = Symbol('t')
    x = Function('x')(t)
    y = Function('y')(t)
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')
    soln = solve([a11 * x + a12 * y - b1, a21 * x + a22 * y - b2], x, y)
    assert soln == {x: (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21), y: (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)}
    assert solve(x - 1, x) == [1]
    assert solve(3 * x - 2, x) == [Rational(2, 3)]
    soln = solve([a11 * x.diff(t) + a12 * y.diff(t) - b1, a21 * x.diff(t) + a22 * y.diff(t) - b2], x.diff(t), y.diff(t))
    assert soln == {y.diff(t): (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21), x.diff(t): (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21)}
    assert solve(x.diff(t) - 1, x.diff(t)) == [1]
    assert solve(3 * x.diff(t) - 2, x.diff(t)) == [Rational(2, 3)]
    eqns = {3 * x - 1, 2 * y - 4}
    assert solve(eqns, {x, y}) == {x: Rational(1, 3), y: 2}
    x = Symbol('x')
    f = Function('f')
    F = x ** 2 + f(x) ** 2 - 4 * x - 1
    assert solve(F.diff(x), diff(f(x), x)) == [(-x + 2) / f(x)]
    x = Symbol('x')
    y = Function('y')(t)
    soln = solve([a11 * x + a12 * y.diff(t) - b1, a21 * x + a22 * y.diff(t) - b2], x, y.diff(t))
    assert soln == {y.diff(t): (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21), x: (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21)}
    x = Symbol('x')
    f = Function('f')
    soln = solve([f(x).diff(x) + f(x).diff(x, 2) - 1, f(x).diff(x) - f(x).diff(x, 2)], f(x).diff(x), f(x).diff(x, 2))
    assert soln == {f(x).diff(x, 2): S(1) / 2, f(x).diff(x): S(1) / 2}
    soln = solve([f(x).diff(x, 2) + f(x).diff(x, 3) - 1, 1 - f(x).diff(x, 2) - f(x).diff(x, 3), 1 - f(x).diff(x, 3)], f(x).diff(x, 2), f(x).diff(x, 3))
    assert soln == {f(x).diff(x, 2): 0, f(x).diff(x, 3): 1}