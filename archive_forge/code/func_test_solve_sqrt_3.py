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
@slow
def test_solve_sqrt_3():
    R = Symbol('R')
    eq = sqrt(2) * R * sqrt(1 / (R + 1)) + (R + 1) * (sqrt(2) * sqrt(1 / (R + 1)) - 1)
    sol = solveset_complex(eq, R)
    fset = [Rational(5, 3) + 4 * sqrt(10) * cos(atan(3 * sqrt(111) / 251) / 3) / 3, -sqrt(10) * cos(atan(3 * sqrt(111) / 251) / 3) / 3 + 40 * re(1 / ((Rational(-1, 2) - sqrt(3) * I / 2) * (Rational(251, 27) + sqrt(111) * I / 9) ** Rational(1, 3))) / 9 + sqrt(30) * sin(atan(3 * sqrt(111) / 251) / 3) / 3 + Rational(5, 3) + I * (-sqrt(30) * cos(atan(3 * sqrt(111) / 251) / 3) / 3 - sqrt(10) * sin(atan(3 * sqrt(111) / 251) / 3) / 3 + 40 * im(1 / ((Rational(-1, 2) - sqrt(3) * I / 2) * (Rational(251, 27) + sqrt(111) * I / 9) ** Rational(1, 3))) / 9)]
    cset = [40 * re(1 / ((Rational(-1, 2) + sqrt(3) * I / 2) * (Rational(251, 27) + sqrt(111) * I / 9) ** Rational(1, 3))) / 9 - sqrt(10) * cos(atan(3 * sqrt(111) / 251) / 3) / 3 - sqrt(30) * sin(atan(3 * sqrt(111) / 251) / 3) / 3 + Rational(5, 3) + I * (40 * im(1 / ((Rational(-1, 2) + sqrt(3) * I / 2) * (Rational(251, 27) + sqrt(111) * I / 9) ** Rational(1, 3))) / 9 - sqrt(10) * sin(atan(3 * sqrt(111) / 251) / 3) / 3 + sqrt(30) * cos(atan(3 * sqrt(111) / 251) / 3) / 3)]
    fs = FiniteSet(*fset)
    cs = ConditionSet(R, Eq(eq, 0), FiniteSet(*cset))
    assert sol == fs - {-1} | cs - {-1}
    eq = -sqrt((m - q) ** 2 + (-m / (2 * q) + S.Half) ** 2) + sqrt((-m ** 2 / 2 - sqrt(4 * m ** 4 - 4 * m ** 2 + 8 * m + 1) / 4 - Rational(1, 4)) ** 2 + (m ** 2 / 2 - m - sqrt(4 * m ** 4 - 4 * m ** 2 + 8 * m + 1) / 4 - Rational(1, 4)) ** 2)
    unsolved_object = ConditionSet(q, Eq(sqrt((m - q) ** 2 + (-m / (2 * q) + S.Half) ** 2) - sqrt((-m ** 2 / 2 - sqrt(4 * m ** 4 - 4 * m ** 2 + 8 * m + 1) / 4 - Rational(1, 4)) ** 2 + (m ** 2 / 2 - m - sqrt(4 * m ** 4 - 4 * m ** 2 + 8 * m + 1) / 4 - Rational(1, 4)) ** 2), 0), S.Reals)
    assert solveset_real(eq, q) == unsolved_object