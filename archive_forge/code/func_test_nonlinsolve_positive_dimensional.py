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
def test_nonlinsolve_positive_dimensional():
    x, y, a, b, c, d = symbols('x, y, a, b, c, d', extended_real=True)
    assert nonlinsolve([x * y, x * y - x], [x, y]) == FiniteSet((0, y))
    system = [a ** 2 + a * c, a - b]
    assert nonlinsolve(system, [a, b]) == FiniteSet((0, 0), (-c, -c))
    eq1 = a + b + c + d
    eq2 = a * b + b * c + c * d + d * a
    eq3 = a * b * c + b * c * d + c * d * a + d * a * b
    eq4 = a * b * c * d - 1
    system = [eq1, eq2, eq3, eq4]
    sol1 = (-1 / d, -d, 1 / d, FiniteSet(d) - FiniteSet(0))
    sol2 = (1 / d, -d, -1 / d, FiniteSet(d) - FiniteSet(0))
    soln = FiniteSet(sol1, sol2)
    assert nonlinsolve(system, [a, b, c, d]) == soln
    assert nonlinsolve([x ** 4 - 3 * x ** 2 + y * x, x * z ** 2, y * z - 1], [x, y, z]) == {(0, 1 / z, z)}