from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
from sympy.printing.octave import octave_code, octave_code as mcode
def test_octave_matrix_elements():
    A = Matrix([[x, 2, x * y]])
    assert mcode(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == 'x.^2 + x.*y + 2'
    A = MatrixSymbol('AA', 1, 3)
    assert mcode(A) == 'AA'
    assert mcode(A[0, 0] ** 2 + sin(A[0, 1]) + A[0, 2]) == 'sin(AA(1, 2)) + AA(1, 1).^2 + AA(1, 3)'
    assert mcode(sum(A)) == 'AA(1, 1) + AA(1, 2) + AA(1, 3)'