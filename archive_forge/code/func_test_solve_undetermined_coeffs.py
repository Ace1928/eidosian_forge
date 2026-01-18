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
def test_solve_undetermined_coeffs():
    assert solve_undetermined_coeffs(a * x ** 2 + b * x ** 2 + b * x + 2 * c * x + c + 1, [a, b, c], x) == {a: -2, b: 2, c: -1}
    assert solve_undetermined_coeffs(a / x + b / (x + 1) - (2 * x + 1) / (x ** 2 + x), [a, b], x) == {a: 1, b: 1}
    assert solve_undetermined_coeffs(((c + 1) * a * x ** 2 + (c + 1) * b * x ** 2 + (c + 1) * b * x + (c + 1) * 2 * c * x + (c + 1) ** 2) / (c + 1), [a, b, c], x) == {a: -2, b: 2, c: -1}
    X, Y, Z = (y, x ** y, y * x ** y)
    eq = a * X + b * Y + c * Z - X - 2 * Y - 3 * Z
    coeffs = (a, b, c)
    syms = (x, y)
    assert solve_undetermined_coeffs(eq, coeffs) == {a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, syms) == {a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, *syms) == {a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(a * x + a - 2, [a]) == []
    assert solve_undetermined_coeffs(a ** 2 * x - 4 * x, [a]) == [{a: -2}, {a: 2}]
    assert solve_undetermined_coeffs(0, [a]) == []
    assert solve_undetermined_coeffs(0, [a], dict=True) == []
    assert solve_undetermined_coeffs(0, [a], set=True) == ([], {})
    assert solve_undetermined_coeffs(1, [a]) == []
    abeq = a * x - 2 * x + b - 3
    s = {b, a}
    assert solve_undetermined_coeffs(abeq, s, x) == {a: 2, b: 3}
    assert solve_undetermined_coeffs(abeq, s, x, set=True) == ([a, b], {(2, 3)})
    assert solve_undetermined_coeffs(sin(a * x) - sin(2 * x), (a,)) is None
    assert solve_undetermined_coeffs(a * x + b * x - 2 * x, (a, b)) == {a: 2 - b}