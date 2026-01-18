from sympy.assumptions import Q
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.numbers import I, Integer, oo, pi, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.common import (ShapeError, NonSquareMatrixError,
from sympy.matrices.matrices import MatrixCalculus
from sympy.matrices import (Matrix, diag, eye,
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten
from sympy.testing.pytest import raises, XFAIL
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray as Array
from sympy.abc import x, y, z
def test_wilkinson():
    wminus, wplus = Matrix.wilkinson(1)
    assert wminus == Matrix([[-1, 1, 0], [1, 0, 1], [0, 1, 1]])
    assert wplus == Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    wminus, wplus = Matrix.wilkinson(3)
    assert wminus == Matrix([[-3, 1, 0, 0, 0, 0, 0], [1, -2, 1, 0, 0, 0, 0], [0, 1, -1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 2, 1], [0, 0, 0, 0, 0, 1, 3]])
    assert wplus == Matrix([[3, 1, 0, 0, 0, 0, 0], [1, 2, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 2, 1], [0, 0, 0, 0, 0, 1, 3]])