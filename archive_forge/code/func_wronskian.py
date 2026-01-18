import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
def wronskian(functions, var, method='bareiss'):
    """
    Compute Wronskian for [] of functions

    ::

                         | f1       f2        ...   fn      |
                         | f1'      f2'       ...   fn'     |
                         |  .        .        .      .      |
        W(f1, ..., fn) = |  .        .         .     .      |
                         |  .        .          .    .      |
                         |  (n)      (n)            (n)     |
                         | D   (f1) D   (f2)  ...  D   (fn) |

    see: https://en.wikipedia.org/wiki/Wronskian

    See Also
    ========

    sympy.matrices.matrices.MatrixCalculus.jacobian
    hessian
    """
    functions = [sympify(f) for f in functions]
    n = len(functions)
    if n == 0:
        return S.One
    W = Matrix(n, n, lambda i, j: functions[i].diff(var, j))
    return W.det(method)