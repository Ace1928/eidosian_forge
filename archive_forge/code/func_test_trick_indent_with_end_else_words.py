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
def test_trick_indent_with_end_else_words():
    t1 = S('endless')
    t2 = S('elsewhere')
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert mcode(pw, inline=False) == 'if (x < 0)\n  endless\nelseif (x <= 1)\n  elsewhere\nelse\n  1\nend'