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
def test_rcode_Piecewise_deep():
    p = rcode(2 * Piecewise((x, x < 1), (x + 1, x < 2), (x ** 2, True)))
    assert p == '2*ifelse(x < 1,x,ifelse(x < 2,x + 1,x^2))'
    expr = x * y * z + x ** 2 + y ** 2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    p = rcode(expr)
    ref = 'x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1'
    assert p == ref
    ref = 'c = x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1;'
    p = rcode(expr, assign_to='c')
    assert p == ref