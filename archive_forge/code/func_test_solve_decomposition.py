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
def test_solve_decomposition():
    n = Dummy('n')
    f1 = exp(3 * x) - 6 * exp(2 * x) + 11 * exp(x) - 6
    f2 = sin(x) ** 2 - 2 * sin(x) + 1
    f3 = sin(x) ** 2 - sin(x)
    f4 = sin(x + 1)
    f5 = exp(x + 2) - 1
    f6 = 1 / log(x)
    f7 = 1 / x
    s1 = ImageSet(Lambda(n, 2 * n * pi), S.Integers)
    s2 = ImageSet(Lambda(n, 2 * n * pi + pi), S.Integers)
    s3 = ImageSet(Lambda(n, 2 * n * pi + pi / 2), S.Integers)
    s4 = ImageSet(Lambda(n, 2 * n * pi - 1), S.Integers)
    s5 = ImageSet(Lambda(n, 2 * n * pi - 1 + pi), S.Integers)
    assert solve_decomposition(f1, x, S.Reals) == FiniteSet(0, log(2), log(3))
    assert dumeq(solve_decomposition(f2, x, S.Reals), s3)
    assert dumeq(solve_decomposition(f3, x, S.Reals), Union(s1, s2, s3))
    assert dumeq(solve_decomposition(f4, x, S.Reals), Union(s4, s5))
    assert solve_decomposition(f5, x, S.Reals) == FiniteSet(-2)
    assert solve_decomposition(f6, x, S.Reals) == S.EmptySet
    assert solve_decomposition(f7, x, S.Reals) == S.EmptySet
    assert solve_decomposition(x, x, Interval(1, 2)) == S.EmptySet