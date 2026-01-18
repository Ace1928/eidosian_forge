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
def test_solve_polynomial1():
    assert solve(3 * x - 2, x) == [Rational(2, 3)]
    assert solve(Eq(3 * x, 2), x) == [Rational(2, 3)]
    assert set(solve(x ** 2 - 1, x)) == {-S.One, S.One}
    assert set(solve(Eq(x ** 2, 1), x)) == {-S.One, S.One}
    assert solve(x - y ** 3, x) == [y ** 3]
    rx = root(x, 3)
    assert solve(x - y ** 3, y) == [rx, -rx / 2 - sqrt(3) * I * rx / 2, -rx / 2 + sqrt(3) * I * rx / 2]
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')
    assert solve([a11 * x + a12 * y - b1, a21 * x + a22 * y - b2], x, y) == {x: (a22 * b1 - a12 * b2) / (a11 * a22 - a12 * a21), y: (a11 * b2 - a21 * b1) / (a11 * a22 - a12 * a21)}
    solution = {x: S.Zero, y: S.Zero}
    assert solve((x - y, x + y), x, y) == solution
    assert solve((x - y, x + y), (x, y)) == solution
    assert solve((x - y, x + y), [x, y]) == solution
    assert set(solve(x ** 3 - 15 * x - 4, x)) == {-2 + 3 ** S.Half, S(4), -2 - 3 ** S.Half}
    assert set(solve((x ** 2 - 1) ** 2 - a, x)) == {sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)), sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))}