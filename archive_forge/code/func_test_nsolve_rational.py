from sympy.core.function import nfloat
from sympy.core.numbers import (Float, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from mpmath import mnorm, mpf
from sympy.solvers import nsolve
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.decorator import conserve_mpmath_dps
def test_nsolve_rational():
    x = symbols('x')
    assert nsolve(x - Rational(1, 3), 0, prec=100) == Rational(1, 3).evalf(100)