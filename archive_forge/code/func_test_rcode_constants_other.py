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
def test_rcode_constants_other():
    assert rcode(2 * GoldenRatio) == 'GoldenRatio = 1.61803398874989;\n2*GoldenRatio'
    assert rcode(2 * Catalan) == 'Catalan = 0.915965594177219;\n2*Catalan'
    assert rcode(2 * EulerGamma) == 'EulerGamma = 0.577215664901533;\n2*EulerGamma'