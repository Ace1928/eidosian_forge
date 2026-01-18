from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set
def test_same_setexprs_are_not_identical():
    a = SetExpr(FiniteSet(0, 1))
    b = SetExpr(FiniteSet(0, 1))
    assert (a + b).set == FiniteSet(0, 1, 2)