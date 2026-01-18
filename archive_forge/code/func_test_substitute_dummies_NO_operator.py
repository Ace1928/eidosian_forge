from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_substitute_dummies_NO_operator():
    i, j = symbols('i j', cls=Dummy)
    assert substitute_dummies(att(i, j) * NO(Fd(i) * F(j)) - att(j, i) * NO(Fd(j) * F(i))) == 0