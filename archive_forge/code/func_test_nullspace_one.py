from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixSubspaces
from sympy.matrices import Matrix
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.solvers import solve
def test_nullspace_one():
    m = SubspaceOnlyMatrix([[1, 2, 0, 2, 5], [-2, -5, 1, -1, -8], [0, -3, 3, 4, 1], [3, 6, 0, -7, 2]])
    basis = m.nullspace()
    assert basis[0] == Matrix([-2, 1, 1, 0, 0])
    assert basis[1] == Matrix([-1, -1, 0, -1, 1])
    assert all((e.is_zero for e in m * basis[0]))
    assert all((e.is_zero for e in m * basis[1]))