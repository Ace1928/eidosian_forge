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
def test_symbolic_matrix_elements():
    n, m = symbols('n,m')
    s1 = BBra([n])
    s2 = BKet([m])
    o = B(0)
    e = apply_operators(s1 * o * s2)
    assert e == sqrt(m) * KroneckerDelta(n, m - 1)