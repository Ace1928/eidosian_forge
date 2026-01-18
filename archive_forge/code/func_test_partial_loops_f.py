from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_partial_loops_f():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))
    j = Idx('j', n)
    (f1, code), (f2, interface) = codegen(('matrix_vector', Eq(y[i], A[i, j] * x[j])), 'F95', 'file', header=False, empty=False)
    expected = 'subroutine matrix_vector(A, m, n, o, p, x, y)\nimplicit none\nINTEGER*4, intent(in) :: m\nINTEGER*4, intent(in) :: n\nINTEGER*4, intent(in) :: o\nINTEGER*4, intent(in) :: p\nREAL*8, intent(in), dimension(1:m, 1:p) :: A\nREAL*8, intent(in), dimension(1:n) :: x\nREAL*8, intent(out), dimension(1:%(iup-ilow)s) :: y\nINTEGER*4 :: i\nINTEGER*4 :: j\ndo i = %(ilow)s, %(iup)s\n   y(i) = 0\nend do\ndo i = %(ilow)s, %(iup)s\n   do j = 1, n\n      y(i) = %(rhs)s + y(i)\n   end do\nend do\nend subroutine\n' % {'rhs': '%(rhs)s', 'iup': str(m - 4), 'ilow': str(1 + o), 'iup-ilow': str(m - 4 - o)}
    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or code == expected % {'rhs': 'x(j)*A(i, j)'}