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
def test_piecewise_solveset():
    eq = Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3
    assert set(solveset_real(eq, x)) == set(FiniteSet(-1, 5))
    absxm3 = Piecewise((x - 3, 0 <= x - 3), (3 - x, 0 > x - 3))
    y = Symbol('y', positive=True)
    assert solveset_real(absxm3 - y, x) == FiniteSet(-y + 3, y + 3)
    f = Piecewise(((x - 2) ** 2, x >= 0), (0, True))
    assert solveset(f, x, domain=S.Reals) == Union(FiniteSet(2), Interval(-oo, 0, True, True))
    assert solveset(Piecewise((x + 1, x > 0), (I, True)) - I, x, S.Reals) == Interval(-oo, 0)
    assert solveset(Piecewise((x - 1, Ne(x, I)), (x, True)), x) == FiniteSet(1)
    g = Piecewise((1, x > 10), (0, True))
    assert solveset(g > 0, x, S.Reals) == Interval.open(10, oo)
    from sympy.logic.boolalg import BooleanTrue
    f = BooleanTrue()
    assert solveset(f, x, domain=Interval(-3, 10)) == Interval(-3, 10)
    f = Piecewise((0, Eq(x, 0)), (x ** 2 / Abs(x), True))
    g = Piecewise((0, Eq(x, pi)), ((x - pi) / sin(x), True))
    assert solveset(f, x, domain=S.Reals) == FiniteSet(0)
    assert solveset(g) == FiniteSet(pi)