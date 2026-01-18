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
@slow
def test_sho():
    n, m = symbols('n,m')
    h_n = Bd(n) * B(n) * (n + S.Half)
    H = Sum(h_n, (n, 0, 5))
    o = H.doit(deep=False)
    b = FixedBosonicBasis(2, 6)
    m = matrix_rep(o, b)
    diag = [1, 2, 3, 3, 4, 5, 4, 5, 6, 7, 5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11]
    for i in range(len(diag)):
        assert diag[i] == m[i, i]