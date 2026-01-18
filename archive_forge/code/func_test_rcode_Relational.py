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
def test_rcode_Relational():
    assert rcode(Eq(x, y)) == 'x == y'
    assert rcode(Ne(x, y)) == 'x != y'
    assert rcode(Le(x, y)) == 'x <= y'
    assert rcode(Lt(x, y)) == 'x < y'
    assert rcode(Gt(x, y)) == 'x > y'
    assert rcode(Ge(x, y)) == 'x >= y'