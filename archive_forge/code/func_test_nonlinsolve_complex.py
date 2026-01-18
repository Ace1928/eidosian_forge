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
def test_nonlinsolve_complex():
    n = Dummy('n')
    assert dumeq(nonlinsolve([exp(x) - sin(y), 1 / y - 3], [x, y]), {(ImageSet(Lambda(n, 2 * n * I * pi + log(sin(Rational(1, 3)))), S.Integers), Rational(1, 3))})
    system = [exp(x) - sin(y), 1 / exp(y) - 3]
    assert dumeq(nonlinsolve(system, [x, y]), {(ImageSet(Lambda(n, I * (2 * n * pi + pi) + log(sin(log(3)))), S.Integers), -log(3)), (ImageSet(Lambda(n, I * (2 * n * pi + arg(sin(2 * n * I * pi - log(3)))) + log(Abs(sin(2 * n * I * pi - log(3))))), S.Integers), ImageSet(Lambda(n, 2 * n * I * pi - log(3)), S.Integers))})
    system = [exp(x) - sin(y), y ** 2 - 4]
    assert dumeq(nonlinsolve(system, [x, y]), {(ImageSet(Lambda(n, I * (2 * n * pi + pi) + log(sin(2))), S.Integers), -2), (ImageSet(Lambda(n, 2 * n * I * pi + log(sin(2))), S.Integers), 2)})
    system = [exp(x) - 2, y ** 2 - 2]
    assert dumeq(nonlinsolve(system, [x, y]), {(log(2), -sqrt(2)), (log(2), sqrt(2)), (ImageSet(Lambda(n, 2 * n * I * pi + log(2)), S.Integers), FiniteSet(-sqrt(2))), (ImageSet(Lambda(n, 2 * n * I * pi + log(2)), S.Integers), FiniteSet(sqrt(2)))})