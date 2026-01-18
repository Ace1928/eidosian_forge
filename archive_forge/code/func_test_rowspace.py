from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixSubspaces
from sympy.matrices import Matrix
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.solvers import solve
def test_rowspace():
    m = SubspaceOnlyMatrix([[1, 2, 0, 2, 5], [-2, -5, 1, -1, -8], [0, -3, 3, 4, 1], [3, 6, 0, -7, 2]])
    basis = m.rowspace()
    assert basis[0] == Matrix([[1, 2, 0, 2, 5]])
    assert basis[1] == Matrix([[0, -1, 1, 3, 2]])
    assert basis[2] == Matrix([[0, 0, 0, 5, 5]])
    assert len(basis) == 3