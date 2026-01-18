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
def test_principal_branch():
    from sympy.functions.elementary.complexes import polar_lift, principal_branch
    p = Symbol('p', positive=True)
    x = Symbol('x')
    neg = Symbol('x', negative=True)
    assert principal_branch(polar_lift(x), p) == principal_branch(x, p)
    assert principal_branch(polar_lift(2 + I), p) == principal_branch(2 + I, p)
    assert principal_branch(2 * x, p) == 2 * principal_branch(x, p)
    assert principal_branch(1, pi) == exp_polar(0)
    assert principal_branch(-1, 2 * pi) == exp_polar(I * pi)
    assert principal_branch(-1, pi) == exp_polar(0)
    assert principal_branch(exp_polar(3 * pi * I) * x, 2 * pi) == principal_branch(exp_polar(I * pi) * x, 2 * pi)
    assert principal_branch(neg * exp_polar(pi * I), 2 * pi) == neg * exp_polar(-I * pi)
    assert principal_branch(exp_polar(-I * pi / 2) / polar_lift(neg), 2 * pi) == exp_polar(-I * pi / 2) / neg
    assert N_equals(principal_branch((1 + I) ** 2, 2 * pi), 2 * I)
    assert N_equals(principal_branch((1 + I) ** 2, 3 * pi), 2 * I)
    assert N_equals(principal_branch((1 + I) ** 2, 1 * pi), 2 * I)
    assert principal_branch(x, I).func is principal_branch
    assert principal_branch(x, -4).func is principal_branch
    assert principal_branch(x, -oo).func is principal_branch
    assert principal_branch(x, zoo).func is principal_branch