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
def test_solve_nonlinear_trans():
    x, y = symbols('x, y', real=True)
    soln1 = FiniteSet((2 * LambertW(y / 2), y))
    soln2 = FiniteSet((-x * sqrt(exp(x)), y), (x * sqrt(exp(x)), y))
    soln3 = FiniteSet((x * exp(x / 2), x))
    soln4 = FiniteSet(2 * LambertW(y / 2), y)
    assert nonlinsolve([x ** 2 - y ** 2 / exp(x)], [x, y]) == soln1
    assert nonlinsolve([x ** 2 - y ** 2 / exp(x)], [y, x]) == soln2
    assert nonlinsolve([x ** 2 - y ** 2 / exp(x)], [y, x]) == soln3
    assert nonlinsolve([x ** 2 - y ** 2 / exp(x)], [x, y]) == soln4