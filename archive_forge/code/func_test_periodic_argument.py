from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, expand)
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.funcmatrix import FunctionMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)
from sympy.matrices import SparseMatrix
from sympy.sets.sets import Interval
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
def test_periodic_argument():
    from sympy.functions.elementary.complexes import periodic_argument, polar_lift, principal_branch, unbranched_argument
    x = Symbol('x')
    p = Symbol('p', positive=True)
    assert unbranched_argument(2 + I) == periodic_argument(2 + I, oo)
    assert unbranched_argument(1 + x) == periodic_argument(1 + x, oo)
    assert N_equals(unbranched_argument((1 + I) ** 2), pi / 2)
    assert N_equals(unbranched_argument((1 - I) ** 2), -pi / 2)
    assert N_equals(periodic_argument((1 + I) ** 2, 3 * pi), pi / 2)
    assert N_equals(periodic_argument((1 - I) ** 2, 3 * pi), -pi / 2)
    assert unbranched_argument(principal_branch(x, pi)) == periodic_argument(x, pi)
    assert unbranched_argument(polar_lift(2 + I)) == unbranched_argument(2 + I)
    assert periodic_argument(polar_lift(2 + I), 2 * pi) == periodic_argument(2 + I, 2 * pi)
    assert periodic_argument(polar_lift(2 + I), 3 * pi) == periodic_argument(2 + I, 3 * pi)
    assert periodic_argument(polar_lift(2 + I), pi) == periodic_argument(polar_lift(2 + I), pi)
    assert unbranched_argument(polar_lift(1 + I)) == pi / 4
    assert periodic_argument(2 * p, p) == periodic_argument(p, p)
    assert periodic_argument(pi * p, p) == periodic_argument(p, p)
    assert Abs(polar_lift(1 + I)) == Abs(1 + I)