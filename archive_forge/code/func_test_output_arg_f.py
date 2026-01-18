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
def test_output_arg_f():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import cos, sin
    x, y, z = symbols('x,y,z')
    r = make_routine('foo', [Equality(y, sin(x)), cos(x)])
    c = FCodeGen()
    result = c.write([r], 'test', header=False, empty=False)
    assert result[0][0] == 'test.f90'
    assert result[0][1] == 'REAL*8 function foo(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(out) :: y\ny = sin(x)\nfoo = cos(x)\nend function\n'