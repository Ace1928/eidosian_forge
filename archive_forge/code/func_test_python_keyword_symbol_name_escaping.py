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
def test_python_keyword_symbol_name_escaping():
    assert python(5 * Symbol('lambda')) == "lambda_ = Symbol('lambda')\ne = 5*lambda_"
    assert python(5 * Symbol('lambda') + 7 * Symbol('lambda_')) == "lambda__ = Symbol('lambda')\nlambda_ = Symbol('lambda_')\ne = 7*lambda_ + 5*lambda__"
    assert python(5 * Symbol('for') + Function('for_')(8)) == "for__ = Symbol('for')\nfor_ = Function('for_')\ne = 5*for__ + for_(8)"