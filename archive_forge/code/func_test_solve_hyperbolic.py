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
def test_solve_hyperbolic():
    n = Dummy('n')
    assert solveset(sinh(x) + cosh(x), x) == S.EmptySet
    assert solveset(sinh(x) + cos(x), x) == ConditionSet(x, Eq(cos(x) + sinh(x), 0), S.Complexes)
    assert solveset_real(sinh(x) + sech(x), x) == FiniteSet(log(sqrt(sqrt(5) - 2)))
    assert solveset_real(3 * cosh(2 * x) - 5, x) == FiniteSet(-log(3) / 2, log(3) / 2)
    assert solveset_real(sinh(x - 3) - 2, x) == FiniteSet(log((2 + sqrt(5)) * exp(3)))
    assert solveset_real(cosh(2 * x) + 2 * sinh(x) - 5, x) == FiniteSet(log(-2 + sqrt(5)), log(1 + sqrt(2)))
    assert solveset_real((coth(x) + sinh(2 * x)) / cosh(x) - 3, x) == FiniteSet(log(S.Half + sqrt(5) / 2), log(1 + sqrt(2)))
    assert solveset_real(cosh(x) * sinh(x) - 2, x) == FiniteSet(log(4 + sqrt(17)) / 2)
    assert solveset_real(sinh(x) + tanh(x) - 1, x) == FiniteSet(log(sqrt(2) / 2 + sqrt(-S(1) / 2 + sqrt(2))))
    assert dumeq(solveset_complex(sinh(x) - I / 2, x), Union(ImageSet(Lambda(n, I * (2 * n * pi + 5 * pi / 6)), S.Integers), ImageSet(Lambda(n, I * (2 * n * pi + pi / 6)), S.Integers)))
    assert dumeq(solveset_complex(sinh(x) + sech(x), x), Union(ImageSet(Lambda(n, 2 * n * I * pi + log(sqrt(-2 + sqrt(5)))), S.Integers), ImageSet(Lambda(n, I * (2 * n * pi + pi / 2) + log(sqrt(2 + sqrt(5)))), S.Integers), ImageSet(Lambda(n, I * (2 * n * pi + pi) + log(sqrt(-2 + sqrt(5)))), S.Integers), ImageSet(Lambda(n, I * (2 * n * pi - pi / 2) + log(sqrt(2 + sqrt(5)))), S.Integers)))
    assert dumeq(solveset(sinh(x / 10) + Rational(3, 4)), Union(ImageSet(Lambda(n, 10 * I * (2 * n * pi + pi) + 10 * log(2)), S.Integers), ImageSet(Lambda(n, 20 * n * I * pi - 10 * log(2)), S.Integers)))
    assert dumeq(solveset(cosh(x / 15) + cosh(x / 5)), Union(ImageSet(Lambda(n, 15 * I * (2 * n * pi + pi / 2)), S.Integers), ImageSet(Lambda(n, 15 * I * (2 * n * pi - pi / 2)), S.Integers), ImageSet(Lambda(n, 15 * I * (2 * n * pi - 3 * pi / 4)), S.Integers), ImageSet(Lambda(n, 15 * I * (2 * n * pi + 3 * pi / 4)), S.Integers), ImageSet(Lambda(n, 15 * I * (2 * n * pi - pi / 4)), S.Integers), ImageSet(Lambda(n, 15 * I * (2 * n * pi + pi / 4)), S.Integers)))
    assert dumeq(solveset(sech(sqrt(2) * x / 3) + 5), Union(ImageSet(Lambda(n, 3 * sqrt(2) * I * (2 * n * pi - pi + atan(2 * sqrt(6))) / 2), S.Integers), ImageSet(Lambda(n, 3 * sqrt(2) * I * (2 * n * pi - atan(2 * sqrt(6)) + pi) / 2), S.Integers)))
    assert dumeq(solveset(tanh(pi * x) - coth(pi / 2 * x)), Union(ImageSet(Lambda(n, 2 * I * (2 * n * pi + pi / 2) / pi), S.Integers), ImageSet(Lambda(n, 2 * I * (2 * n * pi - pi / 2) / pi), S.Integers)))
    assert dumeq(solveset(cosh(9 * x)), Union(ImageSet(Lambda(n, I * (2 * n * pi + pi / 2) / 9), S.Integers), ImageSet(Lambda(n, I * (2 * n * pi - pi / 2) / 9), S.Integers)))
    assert solveset(sinh(x), x, S.Reals) == FiniteSet(0)
    assert dumeq(solveset(sinh(x), x, S.Complexes), Union(ImageSet(Lambda(n, I * (2 * n * pi + pi)), S.Integers), ImageSet(Lambda(n, 2 * n * I * pi), S.Integers)))
    assert dumeq(solveset(sin(pi * x), x, S.Reals), Union(ImageSet(Lambda(n, (2 * n * pi + pi) / pi), S.Integers), ImageSet(Lambda(n, 2 * n), S.Integers)))
    assert dumeq(solveset(sin(pi * x), x), Union(ImageSet(Lambda(n, (2 * n * pi + pi) / pi), S.Integers), ImageSet(Lambda(n, 2 * n), S.Integers)))
    assert dumeq(simplify(solveset(I * cot(8 * x - 8 * E), x)), Union(ImageSet(Lambda(n, n * pi / 4 - 13 * pi / 16 + E), S.Integers), ImageSet(Lambda(n, n * pi / 4 - 11 * pi / 16 + E), S.Integers)))
    assert solveset(cosh(x) + cosh(3 * x) - cosh(5 * x), x, S.Reals).dummy_eq(ConditionSet(x, Eq(cosh(x) + cosh(3 * x) - cosh(5 * x), 0), S.Reals))
    assert solveset(sinh(8 * x) + coth(12 * x)).dummy_eq(ConditionSet(x, Eq(sinh(8 * x) + coth(12 * x), 0), S.Complexes))