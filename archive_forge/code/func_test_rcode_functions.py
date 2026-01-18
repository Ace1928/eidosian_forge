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
def test_rcode_functions():
    assert rcode(sin(x) ** cos(x)) == 'sin(x)^cos(x)'
    assert rcode(factorial(x) + gamma(y)) == 'factorial(x) + gamma(y)'
    assert rcode(beta(Min(x, y), Max(x, y))) == 'beta(min(x, y), max(x, y))'