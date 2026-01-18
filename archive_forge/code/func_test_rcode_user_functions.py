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
def test_rcode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {'ceiling': 'myceil', 'Abs': [(lambda x: not x.is_integer, 'fabs'), (lambda x: x.is_integer, 'abs')]}
    assert rcode(ceiling(x), user_functions=custom_functions) == 'myceil(x)'
    assert rcode(Abs(x), user_functions=custom_functions) == 'fabs(x)'
    assert rcode(Abs(n), user_functions=custom_functions) == 'abs(n)'