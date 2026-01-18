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
def test_python_derivatives():
    f_1 = Derivative(log(x), x, evaluate=False)
    assert python(f_1) == "x = Symbol('x')\ne = Derivative(log(x), x)"
    f_2 = Derivative(log(x), x, evaluate=False) + x
    assert python(f_2) == "x = Symbol('x')\ne = x + Derivative(log(x), x)"
    f_3 = Derivative(log(x) + x ** 2, x, y, evaluate=False)
    assert python(f_3) == "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(x**2 + log(x), x, y)"
    f_4 = Derivative(2 * x * y, y, x, evaluate=False) + x ** 2
    assert python(f_4) in ["x = Symbol('x')\ny = Symbol('y')\ne = x**2 + Derivative(2*x*y, y, x)", "x = Symbol('x')\ny = Symbol('y')\ne = Derivative(2*x*y, y, x) + x**2"]