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
def test_solve_linear():
    w = Wild('w')
    assert solve_linear(x, x) == (0, 1)
    assert solve_linear(x, exclude=[x]) == (0, 1)
    assert solve_linear(x, symbols=[w]) == (0, 1)
    assert solve_linear(x, y - 2 * x) in [(x, y / 3), (y, 3 * x)]
    assert solve_linear(x, y - 2 * x, exclude=[x]) == (y, 3 * x)
    assert solve_linear(3 * x - y, 0) in [(x, y / 3), (y, 3 * x)]
    assert solve_linear(3 * x - y, 0, [x]) == (x, y / 3)
    assert solve_linear(3 * x - y, 0, [y]) == (y, 3 * x)
    assert solve_linear(x ** 2 / y, 1) == (y, x ** 2)
    assert solve_linear(w, x) in [(w, x), (x, w)]
    assert solve_linear(cos(x) ** 2 + sin(x) ** 2 + 2 + y) == (y, -2 - cos(x) ** 2 - sin(x) ** 2)
    assert solve_linear(cos(x) ** 2 + sin(x) ** 2 + 2 + y, symbols=[x]) == (0, 1)
    assert solve_linear(Eq(x, 3)) == (x, 3)
    assert solve_linear(1 / (1 / x - 2)) == (0, 0)
    assert solve_linear((x + 1) * exp(-x), symbols=[x]) == (x, -1)
    assert solve_linear((x + 1) * exp(x), symbols=[x]) == ((x + 1) * exp(x), 1)
    assert solve_linear(x * exp(-x ** 2), symbols=[x]) == (x, 0)
    assert solve_linear(0 ** x - 1) == (0 ** x - 1, 1)
    assert solve_linear(1 + 1 / (x - 1)) == (x, 0)
    eq = y * cos(x) ** 2 + y * sin(x) ** 2 - y
    assert solve_linear(eq) == (0, 1)
    eq = cos(x) ** 2 + sin(x) ** 2
    assert solve_linear(eq) == (0, 1)
    raises(ValueError, lambda: solve_linear(Eq(x, 3), 3))