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
@XFAIL
def test_python_functions_conjugates():
    a, b = map(Symbol, 'ab')
    assert python(conjugate(a + b * I)) == '_     _\na - I*b'
    assert python(conjugate(exp(a + b * I))) == ' _     _\n a - I*b\ne       '