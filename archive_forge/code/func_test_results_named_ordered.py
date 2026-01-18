from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y) * z)
    expr2 = Equality(A, (x - y) * z)
    expr3 = Equality(B, 2 * x)
    name_expr = ('test', [expr1, expr2, expr3])
    result = codegen(name_expr, 'Octave', header=False, empty=False, argument_sequence=(x, z, y))
    assert result[0][0] == 'test.m'
    source = result[0][1]
    expected = 'function [C, A, B] = test(x, z, y)\n  C = z.*(x + y);\n  A = z.*(x - y);\n  B = 2*x;\nend\n'
    assert source == expected