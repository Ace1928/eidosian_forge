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
def test_nonlinsolve_basic():
    assert nonlinsolve([], []) == S.EmptySet
    assert nonlinsolve([], [x, y]) == S.EmptySet
    system = [x, y - x - 5]
    assert nonlinsolve([x], [x, y]) == FiniteSet((0, y))
    assert nonlinsolve(system, [y]) == S.EmptySet
    soln = (ImageSet(Lambda(n, 2 * n * pi + pi / 2), S.Integers),)
    assert dumeq(nonlinsolve([sin(x) - 1], [x]), FiniteSet(tuple(soln)))
    soln = ((ImageSet(Lambda(n, 2 * n * pi + pi), S.Integers), FiniteSet(1)), (ImageSet(Lambda(n, 2 * n * pi), S.Integers), FiniteSet(1)))
    assert dumeq(nonlinsolve([sin(x), y - 1], [x, y]), FiniteSet(*soln))
    assert nonlinsolve([x ** 2 - 1], [x]) == FiniteSet((-1,), (1,))
    soln = FiniteSet((y, y))
    assert nonlinsolve([x - y, 0], x, y) == soln
    assert nonlinsolve([0, x - y], x, y) == soln
    assert nonlinsolve([x - y, x - y], x, y) == soln
    assert nonlinsolve([x, 0], x, y) == FiniteSet((0, y))
    f = Function('f')
    assert nonlinsolve([f(x), 0], f(x), y) == FiniteSet((0, y))
    assert nonlinsolve([f(x), 0], f(x), f(y)) == FiniteSet((0, f(y)))
    A = Indexed('A', x)
    assert nonlinsolve([A, 0], A, y) == FiniteSet((0, y))
    assert nonlinsolve([x ** 2 - 1], [sin(x)]) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x ** 2 - 1], sin(x)) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x ** 2 - 1], 1) == FiniteSet((x ** 2,))
    assert nonlinsolve([x ** 2 - 1], x + y) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([Eq(1, x + y), Eq(1, -x + y - 1), Eq(1, -x + y - 1)], x, y) == FiniteSet((-S.Half, 3 * S.Half))