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
def test_nsolve_dict_kwarg():
    x, y = symbols('x y')
    assert nsolve(x ** 2 - 2, 1, dict=True) == [{x: sqrt(2.0)}]
    assert nsolve(x ** 2 + 2, I, dict=True) == [{x: sqrt(2.0) * I}]
    assert nsolve([x ** 2 + y ** 2 - 5, x ** 2 - y ** 2 + 1], [x, y], [1, 1], dict=True) == [{x: sqrt(2.0), y: sqrt(3.0)}]