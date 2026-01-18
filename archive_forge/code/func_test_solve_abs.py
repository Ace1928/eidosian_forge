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
def test_solve_abs():
    n = Dummy('n')
    raises(ValueError, lambda: solveset(Abs(x) - 1, x))
    assert solveset(Abs(x) - n, x, S.Reals).dummy_eq(ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}))
    assert solveset_real(Abs(x) - 2, x) == FiniteSet(-2, 2)
    assert solveset_real(Abs(x) + 2, x) is S.EmptySet
    assert solveset_real(Abs(x + 3) - 2 * Abs(x - 3), x) == FiniteSet(1, 9)
    assert solveset_real(2 * Abs(x) - Abs(x - 1), x) == FiniteSet(-1, Rational(1, 3))
    sol = ConditionSet(x, And(Contains(b, Interval(0, oo)), Contains(a + b, Interval(0, oo)), Contains(a - b, Interval(0, oo))), FiniteSet(-a - b - 3, -a + b - 3, a - b - 3, a + b - 3))
    eq = Abs(Abs(x + 3) - a) - b
    assert invert_real(eq, 0, x)[1] == sol
    reps = {a: 3, b: 1}
    eqab = eq.subs(reps)
    for si in sol.subs(reps):
        assert not eqab.subs(x, si)
    assert dumeq(solveset(Eq(sin(Abs(x)), 1), x, domain=S.Reals), Union(Intersection(Interval(0, oo), ImageSet(Lambda(n, (-1) ** n * pi / 2 + n * pi), S.Integers)), Intersection(Interval(-oo, 0), ImageSet(Lambda(n, n * pi - (-1) ** (-n) * pi / 2), S.Integers))))