from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.rcode import rcode
def test_rcode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    s = 'for (i in 1:m){\n   y[i] = 0;\n}\nfor (i in 1:m){\n   for (j in 1:n){\n      for (k in 1:o){\n         for (l in 1:p){\n            y[i] = (a[i, j, k, l] + b[i, j, k, l])*c[j, k, l] + y[i];\n         }\n      }\n   }\n}'
    c = rcode((a[i, j, k, l] + b[i, j, k, l]) * c[j, k, l], assign_to=y[i])
    assert c == s