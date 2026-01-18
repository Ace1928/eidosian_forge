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
def test_rcode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    s0 = 'for (i in 1:m){\n   y[i] = 0;\n}\n'
    s1 = 'for (i in 1:m){\n   for (j in 1:n){\n      for (k in 1:o){\n         y[i] = b[j]*b[k]*c[i, j, k] + y[i];\n      }\n   }\n}\n'
    s2 = 'for (i in 1:m){\n   for (k in 1:o){\n      y[i] = a[i, k]*b[k] + y[i];\n   }\n}\n'
    s3 = 'for (i in 1:m){\n   for (j in 1:n){\n      y[i] = a[i, j]*b[j] + y[i];\n   }\n}\n'
    c = rcode(b[j] * a[i, j] + b[k] * a[i, k] + b[j] * b[k] * c[i, j, k], assign_to=y[i])
    ref = {}
    ref[0] = s0 + s1 + s2 + s3[:-1]
    ref[1] = s0 + s1 + s3 + s2[:-1]
    ref[2] = s0 + s2 + s1 + s3[:-1]
    ref[3] = s0 + s2 + s3 + s1[:-1]
    ref[4] = s0 + s3 + s1 + s2[:-1]
    ref[5] = s0 + s3 + s2 + s1[:-1]
    assert c == ref[0] or c == ref[1] or c == ref[2] or (c == ref[3]) or (c == ref[4]) or (c == ref[5])