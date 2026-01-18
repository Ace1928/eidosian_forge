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
def test_row_insert():
    r4 = Matrix([[4, 4, 4]])
    for i in range(-4, 5):
        l = [1, 0, 0]
        l.insert(i, 4)
        assert flatten(eye_Shaping(3).row_insert(i, r4).col(0).tolist()) == l