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
@_both_exp_pow
def test_polarify():
    from sympy.functions.elementary.complexes import polar_lift, polarify
    x = Symbol('x')
    z = Symbol('z', polar=True)
    f = Function('f')
    ES = {}
    assert polarify(-1) == (polar_lift(-1), ES)
    assert polarify(1 + I) == (polar_lift(1 + I), ES)
    assert polarify(exp(x), subs=False) == exp(x)
    assert polarify(1 + x, subs=False) == 1 + x
    assert polarify(f(I) + x, subs=False) == f(polar_lift(I)) + x
    assert polarify(x, lift=True) == polar_lift(x)
    assert polarify(z, lift=True) == z
    assert polarify(f(x), lift=True) == f(polar_lift(x))
    assert polarify(1 + x, lift=True) == polar_lift(1 + x)
    assert polarify(1 + f(x), lift=True) == polar_lift(1 + f(polar_lift(x)))
    newex, subs = polarify(f(x) + z)
    assert newex.subs(subs) == f(x) + z
    mu = Symbol('mu')
    sigma = Symbol('sigma', positive=True)
    assert polarify(Integral(sqrt(2) * x * exp(-(-mu + x) ** 2 / (2 * sigma ** 2)) / (2 * sqrt(pi) * sigma), (x, -oo, oo)), lift=True) == Integral(sqrt(2) * (sigma * exp_polar(0)) ** exp_polar(I * pi) * exp((sigma * exp_polar(0)) ** (2 * exp_polar(I * pi)) * exp_polar(I * pi) * polar_lift(-mu + x) ** (2 * exp_polar(0)) / 2) * exp_polar(0) * polar_lift(x) / (2 * sqrt(pi)), (x, -oo, oo))