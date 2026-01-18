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
def test_substitute_dummies_substitution_order():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    f = Function('f')
    from sympy.utilities.iterables import variations
    for permut in variations([i, j, k, l], 4):
        assert substitute_dummies(f(*permut) - f(i, j, k, l)) == 0