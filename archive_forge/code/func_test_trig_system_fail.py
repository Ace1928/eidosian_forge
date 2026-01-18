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
@XFAIL
def test_trig_system_fail():
    sys = [x + y - pi / 2, sin(x) + sin(y) - 1]
    soln_1 = (ImageSet(Lambda(n, n * pi + pi / 2), S.Integers), ImageSet(Lambda(n, n * pi), S.Integers))
    soln_1 = FiniteSet(soln_1)
    soln_2 = (ImageSet(Lambda(n, n * pi), S.Integers), ImageSet(Lambda(n, n * pi + pi / 2), S.Integers))
    soln_2 = FiniteSet(soln_2)
    soln = soln_1 + soln_2
    assert dumeq(nonlinsolve(sys, [x, y]), soln)
    sys = [sin(x) + sin(y) - (sqrt(3) + 1) / 2, sin(x) - sin(y) - (sqrt(3) - 1) / 2]
    soln_x = Union(ImageSet(Lambda(n, 2 * n * pi + pi / 3), S.Integers), ImageSet(Lambda(n, 2 * n * pi + pi * Rational(2, 3)), S.Integers))
    soln_y = Union(ImageSet(Lambda(n, 2 * n * pi + pi / 6), S.Integers), ImageSet(Lambda(n, 2 * n * pi + pi * Rational(5, 6)), S.Integers))
    assert dumeq(nonlinsolve(sys, [x, y]), FiniteSet((soln_x, soln_y)))