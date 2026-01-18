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
@_both_exp_pow
def test_solve_trig():
    assert dumeq(solveset_real(sin(x), x), Union(imageset(Lambda(n, 2 * pi * n), S.Integers), imageset(Lambda(n, 2 * pi * n + pi), S.Integers)))
    assert dumeq(solveset_real(sin(x) - 1, x), imageset(Lambda(n, 2 * pi * n + pi / 2), S.Integers))
    assert dumeq(solveset_real(cos(x), x), Union(imageset(Lambda(n, 2 * pi * n + pi / 2), S.Integers), imageset(Lambda(n, 2 * pi * n + pi * Rational(3, 2)), S.Integers)))
    assert dumeq(solveset_real(sin(x) + cos(x), x), Union(imageset(Lambda(n, 2 * n * pi + pi * Rational(3, 4)), S.Integers), imageset(Lambda(n, 2 * n * pi + pi * Rational(7, 4)), S.Integers)))
    assert solveset_real(sin(x) ** 2 + cos(x) ** 2, x) == S.EmptySet
    assert dumeq(solveset_complex(cos(x) - S.Half, x), Union(imageset(Lambda(n, 2 * n * pi + pi * Rational(5, 3)), S.Integers), imageset(Lambda(n, 2 * n * pi + pi / 3), S.Integers)))
    assert dumeq(solveset(sin(y + a) - sin(y), a, domain=S.Reals), Union(ImageSet(Lambda(n, 2 * n * pi), S.Integers), Intersection(ImageSet(Lambda(n, -I * (I * (2 * n * pi + arg(-exp(-2 * I * y))) + 2 * im(y))), S.Integers), S.Reals)))
    assert dumeq(solveset_real(sin(2 * x) * cos(x) + cos(2 * x) * sin(x) - 1, x), ImageSet(Lambda(n, n * pi * Rational(2, 3) + pi / 6), S.Integers))
    assert dumeq(solveset_real(2 * tan(x) * sin(x) + 1, x), Union(ImageSet(Lambda(n, 2 * n * pi + atan(sqrt(2) * sqrt(-1 + sqrt(17)) / (1 - sqrt(17))) + pi), S.Integers), ImageSet(Lambda(n, 2 * n * pi - atan(sqrt(2) * sqrt(-1 + sqrt(17)) / (1 - sqrt(17))) + pi), S.Integers)))
    assert dumeq(solveset_real(cos(2 * x) * cos(4 * x) - 1, x), ImageSet(Lambda(n, n * pi), S.Integers))
    assert dumeq(solveset(sin(x / 10) + Rational(3, 4)), Union(ImageSet(Lambda(n, 20 * n * pi + 10 * atan(3 * sqrt(7) / 7) + 10 * pi), S.Integers), ImageSet(Lambda(n, 20 * n * pi - 10 * atan(3 * sqrt(7) / 7) + 20 * pi), S.Integers)))
    assert dumeq(solveset(cos(x / 15) + cos(x / 5)), Union(ImageSet(Lambda(n, 30 * n * pi + 15 * pi / 2), S.Integers), ImageSet(Lambda(n, 30 * n * pi + 45 * pi / 2), S.Integers), ImageSet(Lambda(n, 30 * n * pi + 75 * pi / 4), S.Integers), ImageSet(Lambda(n, 30 * n * pi + 45 * pi / 4), S.Integers), ImageSet(Lambda(n, 30 * n * pi + 105 * pi / 4), S.Integers), ImageSet(Lambda(n, 30 * n * pi + 15 * pi / 4), S.Integers)))
    assert dumeq(solveset(sec(sqrt(2) * x / 3) + 5), Union(ImageSet(Lambda(n, 3 * sqrt(2) * (2 * n * pi - pi + atan(2 * sqrt(6))) / 2), S.Integers), ImageSet(Lambda(n, 3 * sqrt(2) * (2 * n * pi - atan(2 * sqrt(6)) + pi) / 2), S.Integers)))
    assert dumeq(simplify(solveset(tan(pi * x) - cot(pi / 2 * x))), Union(ImageSet(Lambda(n, 4 * n + 1), S.Integers), ImageSet(Lambda(n, 4 * n + 3), S.Integers), ImageSet(Lambda(n, 4 * n + Rational(7, 3)), S.Integers), ImageSet(Lambda(n, 4 * n + Rational(5, 3)), S.Integers), ImageSet(Lambda(n, 4 * n + Rational(11, 3)), S.Integers), ImageSet(Lambda(n, 4 * n + Rational(1, 3)), S.Integers)))
    assert dumeq(solveset(cos(9 * x)), Union(ImageSet(Lambda(n, 2 * n * pi / 9 + pi / 18), S.Integers), ImageSet(Lambda(n, 2 * n * pi / 9 + pi / 6), S.Integers)))
    assert dumeq(solveset(sin(8 * x) + cot(12 * x), x, S.Reals), Union(ImageSet(Lambda(n, n * pi / 2 + pi / 8), S.Integers), ImageSet(Lambda(n, n * pi / 2 + 3 * pi / 8), S.Integers), ImageSet(Lambda(n, n * pi / 2 + 5 * pi / 16), S.Integers), ImageSet(Lambda(n, n * pi / 2 + 3 * pi / 16), S.Integers), ImageSet(Lambda(n, n * pi / 2 + 7 * pi / 16), S.Integers), ImageSet(Lambda(n, n * pi / 2 + pi / 16), S.Integers)))
    assert dumeq(solveset_real(2 * cos(x) * cos(2 * x) - 1, x), Union(ImageSet(Lambda(n, 2 * n * pi + 2 * atan(sqrt(-2 * 2 ** Rational(1, 3) * (67 + 9 * sqrt(57)) ** Rational(2, 3) + 8 * 2 ** Rational(2, 3) + 11 * (67 + 9 * sqrt(57)) ** Rational(1, 3)) / (3 * (67 + 9 * sqrt(57)) ** Rational(1, 6)))), S.Integers), ImageSet(Lambda(n, 2 * n * pi - 2 * atan(sqrt(-2 * 2 ** Rational(1, 3) * (67 + 9 * sqrt(57)) ** Rational(2, 3) + 8 * 2 ** Rational(2, 3) + 11 * (67 + 9 * sqrt(57)) ** Rational(1, 3)) / (3 * (67 + 9 * sqrt(57)) ** Rational(1, 6))) + 2 * pi), S.Integers)))
    assert dumeq(simplify(solveset(sin(x / 180 * pi) - S.Half, x, S.Reals)), Union(ImageSet(Lambda(n, 360 * n + 150), S.Integers), ImageSet(Lambda(n, 360 * n + 30), S.Integers)))