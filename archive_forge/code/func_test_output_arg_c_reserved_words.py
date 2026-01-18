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
def test_output_arg_c_reserved_words():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import cos, sin
    x, y, z = symbols('if, while, z')
    r = make_routine('foo', [Equality(y, sin(x)), cos(x)])
    c = C89CodeGen()
    result = c.write([r], 'test', header=False, empty=False)
    assert result[0][0] == 'test.c'
    expected = '#include "test.h"\n#include <math.h>\ndouble foo(double if_, double *while_) {\n   (*while_) = sin(if_);\n   double foo_result;\n   foo_result = cos(if_);\n   return foo_result;\n}\n'
    assert result[0][1] == expected