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
def test_spread_assign_to_deeply_nested_symbols():
    gl = glsl_code
    a, b, c, x, y, z = symbols('a b c x y z')
    expr = (((1, 2), 3), ((1, 2), 3))
    assign_to = (((a, b), c), ((x, y), z))
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('        a = 1;\n        b = 2;\n        c = 3;\n        x = 1;\n        y = 2;\n        z = 3;')