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
def test_solve_trig_hyp_symbolic():
    assert dumeq(solveset(sin(a * x), x), ConditionSet(x, Ne(a, 0), Union(ImageSet(Lambda(n, (2 * n * pi + pi) / a), S.Integers), ImageSet(Lambda(n, 2 * n * pi / a), S.Integers))))
    assert dumeq(solveset(cosh(x / a), x), ConditionSet(x, Ne(a, 0), Union(ImageSet(Lambda(n, I * a * (2 * n * pi + pi / 2)), S.Integers), ImageSet(Lambda(n, I * a * (2 * n * pi - pi / 2)), S.Integers))))
    assert dumeq(solveset(sin(2 * sqrt(3) / 3 * a ** 2 / (b * pi) * x) + cos(4 * sqrt(3) / 3 * a ** 2 / (b * pi) * x), x), ConditionSet(x, Ne(b, 0) & Ne(a ** 2, 0), Union(ImageSet(Lambda(n, sqrt(3) * pi * b * (2 * n * pi + pi / 2) / (2 * a ** 2)), S.Integers), ImageSet(Lambda(n, sqrt(3) * pi * b * (2 * n * pi - 5 * pi / 6) / (2 * a ** 2)), S.Integers), ImageSet(Lambda(n, sqrt(3) * pi * b * (2 * n * pi - pi / 6) / (2 * a ** 2)), S.Integers))))
    assert dumeq(simplify(solveset(cot((1 + I) * x) - cot((3 + 3 * I) * x), x)), Union(ImageSet(Lambda(n, pi * (1 - I) * (4 * n + 1) / 4), S.Integers), ImageSet(Lambda(n, pi * (1 - I) * (4 * n - 1) / 4), S.Integers)))
    assert dumeq(solveset(cosh((a ** 2 + 1) * x) - 3, x), ConditionSet(x, Ne(a ** 2 + 1, 0), Union(ImageSet(Lambda(n, (2 * n * I * pi + log(3 - 2 * sqrt(2))) / (a ** 2 + 1)), S.Integers), ImageSet(Lambda(n, (2 * n * I * pi + log(2 * sqrt(2) + 3)) / (a ** 2 + 1)), S.Integers))))
    ar = Symbol('ar', real=True)
    assert solveset(cosh((ar ** 2 + 1) * x) - 2, x, S.Reals) == FiniteSet(log(sqrt(3) + 2) / (ar ** 2 + 1), log(2 - sqrt(3)) / (ar ** 2 + 1))