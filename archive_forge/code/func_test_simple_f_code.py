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
def test_simple_f_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y) * z
    routine = make_routine('test', expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = 'REAL*8 function test(x, y, z)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nREAL*8, intent(in) :: z\ntest = z*(x + y)\nend function\n'
    assert source == expected