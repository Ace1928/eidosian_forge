from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose
def test_rank_decomposition():
    a = Matrix(0, 0, [])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a
    a = Matrix(1, 1, [5])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a
    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a
    a = Matrix([[0, 0, 1, 2, 2, -5, 3], [-1, 5, 2, 2, 1, -7, 5], [0, 0, -2, -3, -3, 8, -5], [-1, 5, 0, -1, -2, 1, 0]])
    c, f = a.rank_decomposition()
    assert f.is_echelon
    assert c.cols == f.rows == a.rank()
    assert c * f == a