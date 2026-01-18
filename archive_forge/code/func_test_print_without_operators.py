from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap
def test_print_without_operators():
    assert glsl_code(x * y, use_operators=False) == 'mul(x, y)'
    assert glsl_code(x ** y + z, use_operators=False) == 'add(pow(x, y), z)'
    assert glsl_code(x * (y + z), use_operators=False) == 'mul(x, add(y, z))'
    assert glsl_code(x * (y + z), use_operators=False) == 'mul(x, add(y, z))'
    assert glsl_code(x * (y + z ** y ** 0.5), use_operators=False) == 'mul(x, add(y, pow(z, sqrt(y))))'
    assert glsl_code(-x - y, use_operators=False, zero='zero()') == 'sub(zero(), add(x, y))'
    assert glsl_code(-x - y, use_operators=False) == 'sub(0.0, add(x, y))'