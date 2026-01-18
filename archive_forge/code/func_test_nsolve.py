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
def test_nsolve():
    x = Symbol('x')
    assert nsolve(sin(x), 2) - pi.evalf() < 1e-15
    assert nsolve(Eq(2 * x, 2), x, -10) == nsolve(2 * x - 2, -10)
    raises(TypeError, lambda: nsolve(Eq(2 * x, 2)))
    raises(TypeError, lambda: nsolve(Eq(2 * x, 2), x, 1, 2))
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    f1 = 3 * x1 ** 2 - 2 * x2 ** 2 - 1
    f2 = x1 ** 2 - 2 * x1 + x2 ** 2 + 2 * x2 - 8
    f = Matrix((f1, f2)).T
    F = lambdify((x1, x2), f.T, modules='mpmath')
    for x0 in [(-1, 1), (1, -2), (4, 4), (-4, -4)]:
        x = nsolve(f, (x1, x2), x0, tol=1e-08)
        assert mnorm(F(*x), 1) <= 1e-10
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    f1 = -x + 2 * y
    f2 = (x ** 2 + x * (y ** 2 - 2) - 4 * y) / (x + 4)
    f3 = sqrt(x ** 2 + y ** 2) * z
    f = Matrix((f1, f2, f3)).T
    F = lambdify((x, y, z), f.T, modules='mpmath')

    def getroot(x0):
        root = nsolve(f, (x, y, z), x0)
        assert mnorm(F(*root), 1) <= 1e-08
        return root
    assert list(map(round, getroot((1, 1, 1)))) == [2, 1, 0]
    assert nsolve([Eq(f1, 0), Eq(f2, 0), Eq(f3, 0)], [x, y, z], (1, 1, 1))
    a = Symbol('a')
    assert abs(nsolve(1 / (0.001 + a) ** 3 - 6 / (0.9 - a) ** 3, a, 0.3) - mpf('0.31883011387318591')) < 1e-15