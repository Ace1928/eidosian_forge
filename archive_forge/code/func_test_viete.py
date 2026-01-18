from sympy.polys.polyfuncs import (
from sympy.polys.polyerrors import (
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, d, e, x, y, z
def test_viete():
    r1, r2 = symbols('r1, r2')
    assert viete(a * x ** 2 + b * x + c, [r1, r2], x) == [(r1 + r2, -b / a), (r1 * r2, c / a)]
    raises(ValueError, lambda: viete(1, [], x))
    raises(ValueError, lambda: viete(x ** 2 + 1, [r1]))
    raises(MultivariatePolynomialError, lambda: viete(x + y, [r1]))