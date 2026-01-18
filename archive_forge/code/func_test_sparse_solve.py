from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_sparse_solve():
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    assert A.cholesky() == Matrix([[5, 0, 0], [3, 3, 0], [-1, 1, 3]])
    assert A.cholesky() * A.cholesky().T == Matrix([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
    A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    L, D = A.LDLdecomposition()
    assert 15 * L == Matrix([[15, 0, 0], [9, 15, 0], [-3, 5, 15]])
    assert D == Matrix([[25, 0, 0], [0, 9, 0], [0, 0, 9]])
    assert L * D * L.T == A
    A = SparseMatrix(((3, 0, 2), (0, 0, 1), (1, 2, 0)))
    assert A.inv() * A == SparseMatrix(eye(3))
    A = SparseMatrix([[2, -1, 0], [-1, 2, -1], [0, 0, 2]])
    ans = SparseMatrix([[Rational(2, 3), Rational(1, 3), Rational(1, 6)], [Rational(1, 3), Rational(2, 3), Rational(1, 3)], [0, 0, S.Half]])
    assert A.inv(method='CH') == ans
    assert A.inv(method='LDL') == ans
    assert A * ans == SparseMatrix(eye(3))
    s = A.solve(A[:, 0], 'LDL')
    assert A * s == A[:, 0]
    s = A.solve(A[:, 0], 'CH')
    assert A * s == A[:, 0]
    A = A.col_join(A)
    s = A.solve_least_squares(A[:, 0], 'CH')
    assert A * s == A[:, 0]
    s = A.solve_least_squares(A[:, 0], 'LDL')
    assert A * s == A[:, 0]