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
def test_numbersymbol_c_code():
    routine = make_routine('test', pi ** Catalan)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = '#include "file.h"\n#include <math.h>\ndouble test() {\n   double test_result;\n   double const Catalan = %s;\n   test_result = pow(M_PI, Catalan);\n   return test_result;\n}\n' % Catalan.evalf(17)
    assert source == expected