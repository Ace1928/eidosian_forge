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
@XFAIL
def test_principal_branch_fail():
    from sympy.functions.elementary.complexes import principal_branch
    assert N_equals(principal_branch((1 + I) ** 2, pi / 2), 0)