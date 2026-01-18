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
def test_python_relational():
    assert python(Eq(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = Eq(x, y)"
    assert python(Ge(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x >= y"
    assert python(Le(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x <= y"
    assert python(Gt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x > y"
    assert python(Lt(x, y)) == "x = Symbol('x')\ny = Symbol('y')\ne = x < y"
    assert python(Ne(x / (y + 1), y ** 2)) in ["x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(1 + y), y**2)", "x = Symbol('x')\ny = Symbol('y')\ne = Ne(x/(y + 1), y**2)"]