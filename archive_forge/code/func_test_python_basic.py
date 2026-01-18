from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.series.limits import limit
from sympy.printing.python import python
from sympy.testing.pytest import raises, XFAIL
def test_python_basic():
    assert python(-Rational(1) / 2) == 'e = Rational(-1, 2)'
    assert python(-Rational(13) / 22) == 'e = Rational(-13, 22)'
    assert python(oo) == 'e = oo'
    assert python(x ** 2) == "x = Symbol('x')\ne = x**2"
    assert python(1 / x) == "x = Symbol('x')\ne = 1/x"
    assert python(y * x ** (-2)) == "y = Symbol('y')\nx = Symbol('x')\ne = y/x**2"
    assert python(x ** Rational(-5, 2)) == "x = Symbol('x')\ne = x**Rational(-5, 2)"
    assert python(x ** 2 + x + 1) in ["x = Symbol('x')\ne = 1 + x + x**2", "x = Symbol('x')\ne = x + x**2 + 1", "x = Symbol('x')\ne = x**2 + x + 1"]
    assert python(1 - x) in ["x = Symbol('x')\ne = 1 - x", "x = Symbol('x')\ne = -x + 1"]
    assert python(1 - 2 * x) in ["x = Symbol('x')\ne = 1 - 2*x", "x = Symbol('x')\ne = -2*x + 1"]
    assert python(1 - Rational(3, 2) * y / x) in ["y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3/2*y/x", "y = Symbol('y')\nx = Symbol('x')\ne = -3/2*y/x + 1", "y = Symbol('y')\nx = Symbol('x')\ne = 1 - 3*y/(2*x)"]
    assert python(x / y) == "x = Symbol('x')\ny = Symbol('y')\ne = x/y"
    assert python(-x / y) == "x = Symbol('x')\ny = Symbol('y')\ne = -x/y"
    assert python((x + 2) / y) in ["y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(2 + x)", "y = Symbol('y')\nx = Symbol('x')\ne = 1/y*(x + 2)", "x = Symbol('x')\ny = Symbol('y')\ne = 1/y*(2 + x)", "x = Symbol('x')\ny = Symbol('y')\ne = (2 + x)/y", "x = Symbol('x')\ny = Symbol('y')\ne = (x + 2)/y"]
    assert python((1 + x) * y) in ["y = Symbol('y')\nx = Symbol('x')\ne = y*(1 + x)", "y = Symbol('y')\nx = Symbol('x')\ne = y*(x + 1)"]
    assert python(-5 * x / (x + 10)) == "x = Symbol('x')\ne = -5*x/(x + 10)"
    assert python(1 - Rational(3, 2) * (x + 1)) in ["x = Symbol('x')\ne = Rational(-3, 2)*x + Rational(-1, 2)", "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)", "x = Symbol('x')\ne = -3*x/2 + Rational(-1, 2)"]