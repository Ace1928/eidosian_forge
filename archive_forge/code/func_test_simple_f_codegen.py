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
def test_simple_f_codegen():
    x, y, z = symbols('x,y,z')
    expr = (x + y) * z
    result = codegen(('test', expr), 'F95', 'file', header=False, empty=False)
    expected = [('file.f90', 'REAL*8 function test(x, y, z)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nREAL*8, intent(in) :: z\ntest = z*(x + y)\nend function\n'), ('file.h', 'interface\nREAL*8 function test(x, y, z)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nREAL*8, intent(in) :: z\nend function\nend interface\n')]
    assert result == expected