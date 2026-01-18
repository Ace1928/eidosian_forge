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
def test_solve_modular():
    n = Dummy('n', integer=True)
    assert solveset(Mod(x, 4) - x, x, S.Integers).dummy_eq(ConditionSet(x, Eq(-x + Mod(x, 4), 0), S.Integers))
    assert solveset(3 - Mod(sin(x), 7), x, S.Integers).dummy_eq(ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(log(x), 7), x, S.Integers).dummy_eq(ConditionSet(x, Eq(Mod(log(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(exp(x), 7), x, S.Integers).dummy_eq(ConditionSet(x, Eq(Mod(exp(x), 7) - 3, 0), S.Integers))
    assert solveset(7 - Mod(x, 5), x, S.Integers) is S.EmptySet
    assert solveset(5 - Mod(x, 5), x, S.Integers) is S.EmptySet
    assert dumeq(solveset(2 + Mod(x, -3), x, S.Integers), ImageSet(Lambda(n, -3 * n - 2), S.Integers))
    assert solveset(4 + Mod(x, -3), x, S.Integers) is S.EmptySet
    assert dumeq(solveset(3 - Mod(x, 5), x, S.Integers), ImageSet(Lambda(n, 5 * n + 3), S.Integers))
    assert dumeq(solveset(3 - Mod(5 * x - 8, 7), x, S.Integers), ImageSet(Lambda(n, 7 * n + 5), S.Integers))
    assert dumeq(solveset(3 - Mod(5 * x, 7), x, S.Integers), ImageSet(Lambda(n, 7 * n + 2), S.Integers))
    assert dumeq(solveset(Mod(x ** 2, 160) - 9, x, S.Integers), Union(ImageSet(Lambda(n, 160 * n + 3), S.Integers), ImageSet(Lambda(n, 160 * n + 13), S.Integers), ImageSet(Lambda(n, 160 * n + 67), S.Integers), ImageSet(Lambda(n, 160 * n + 77), S.Integers), ImageSet(Lambda(n, 160 * n + 83), S.Integers), ImageSet(Lambda(n, 160 * n + 93), S.Integers), ImageSet(Lambda(n, 160 * n + 147), S.Integers), ImageSet(Lambda(n, 160 * n + 157), S.Integers)))
    assert solveset(3 - Mod(x ** 4, 7), x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x ** 4, 17) - 13, x, S.Integers), Union(ImageSet(Lambda(n, 17 * n + 3), S.Integers), ImageSet(Lambda(n, 17 * n + 5), S.Integers), ImageSet(Lambda(n, 17 * n + 12), S.Integers), ImageSet(Lambda(n, 17 * n + 14), S.Integers)))
    assert dumeq(solveset(Mod(7 ** x, 41) - 15, x, S.Integers), ImageSet(Lambda(n, 40 * n + 3), S.Naturals0))
    assert dumeq(solveset(Mod(12 ** x, 21) - 18, x, S.Integers), ImageSet(Lambda(n, 6 * n + 2), S.Naturals0))
    assert dumeq(solveset(Mod(3 ** x, 4) - 3, x, S.Integers), ImageSet(Lambda(n, 2 * n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(2 ** x, 7) - 2, x, S.Integers), ImageSet(Lambda(n, 3 * n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(3 ** 3 ** x, 4) - 3, x, S.Integers), Intersection(ImageSet(Lambda(n, Intersection({log(2 * n + 1) / log(3)}, S.Integers)), S.Naturals0), S.Integers))
    assert solveset(Mod(x ** 3, 7) - 2, x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x ** 3, 8) - 1, x, S.Integers), ImageSet(Lambda(n, 8 * n + 1), S.Integers))
    assert dumeq(solveset(Mod(x ** 4, 9) - 4, x, S.Integers), Union(ImageSet(Lambda(n, 9 * n + 4), S.Integers), ImageSet(Lambda(n, 9 * n + 5), S.Integers)))
    assert dumeq(solveset(3 - Mod(5 * x - 8, 7), x, S.Naturals0), Intersection(ImageSet(Lambda(n, 7 * n + 5), S.Integers), S.Naturals0))
    assert solveset(Mod(x, 3) - I, x, S.Integers) == S.EmptySet
    assert solveset(Mod(I * x, 3) - 2, x, S.Integers).dummy_eq(ConditionSet(x, Eq(Mod(I * x, 3) - 2, 0), S.Integers))
    assert solveset(Mod(I + x, 3) - 2, x, S.Integers).dummy_eq(ConditionSet(x, Eq(Mod(x + I, 3) - 2, 0), S.Integers))
    assert dumeq(solveset(Mod(x ** 4, 14) - 11, x, S.Integers), Union(ImageSet(Lambda(n, 14 * n + 3), S.Integers), ImageSet(Lambda(n, 14 * n + 11), S.Integers)))
    assert dumeq(solveset(Mod(x ** 31, 74) - 43, x, S.Integers), ImageSet(Lambda(n, 74 * n + 31), S.Integers))
    n = symbols('n', integer=True)
    a = 742938285
    b = 1898888478
    m = 2 ** 31 - 1
    c = 20170816
    assert dumeq(solveset(c - Mod(a ** n * b, m), n, S.Integers), ImageSet(Lambda(n, 2147483646 * n + 100), S.Naturals0))
    assert dumeq(solveset(c - Mod(a ** n * b, m), n, S.Naturals0), Intersection(ImageSet(Lambda(n, 2147483646 * n + 100), S.Naturals0), S.Naturals0))
    assert dumeq(solveset(c - Mod(a ** (2 * n) * b, m), n, S.Integers), Intersection(ImageSet(Lambda(n, 1073741823 * n + 50), S.Naturals0), S.Integers))
    assert solveset(c - Mod(a ** (2 * n + 7) * b, m), n, S.Integers) is S.EmptySet
    assert dumeq(solveset(c - Mod(a ** (n - 4) * b, m), n, S.Integers), Intersection(ImageSet(Lambda(n, 2147483646 * n + 104), S.Naturals0), S.Integers))