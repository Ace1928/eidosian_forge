from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
def test_nonlinsolve_inexact():
    sol = [(-1.625, -1.375), (1.625, 1.375)]
    res = nonlinsolve([(x + y) ** 2 - 9, x ** 2 - y ** 2 - 0.75], [x, y])
    assert all((abs(res.args[i][j] - sol[i][j]) < 1e-09 for i in range(2) for j in range(2)))
    assert nonlinsolve([(x + y) ** 2 - 9, (x + y) ** 2 - 0.75], [x, y]) == S.EmptySet
    assert nonlinsolve([y ** 2 + (x - 0.5) ** 2 - 0.0625, 2 * x - 1.0, 2 * y], [x, y]) == S.EmptySet
    res = nonlinsolve([x ** 2 + y - 0.5, (x + y) ** 2, log(z)], [x, y, z])
    sol = [(-0.366025403784439, 0.366025403784439, 1), (-0.366025403784439, 0.366025403784439, 1), (1.36602540378444, -1.36602540378444, 1)]
    assert all((abs(res.args[i][j] - sol[i][j]) < 1e-09 for i in range(3) for j in range(3)))
    res = nonlinsolve([y - x ** 2, x ** 5 - x + 1.0], [x, y])
    sol = [(-1.16730397826142, 1.36259857766493), (-0.181232444469876 - 1.08395410131771 * I, -1.14211129483496 + 0.392895302949911 * I), (-0.181232444469876 + 1.08395410131771 * I, -1.14211129483496 - 0.392895302949911 * I), (0.764884433600585 - 0.352471546031726 * I, 0.460812006002492 - 0.539199997693599 * I), (0.764884433600585 + 0.352471546031726 * I, 0.460812006002492 + 0.539199997693599 * I)]
    assert all((abs(res.args[i][j] - sol[i][j]) < 1e-09 for i in range(5) for j in range(2)))