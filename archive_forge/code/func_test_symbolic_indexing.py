from copy import copy
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.core.containers import Dict
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices import SparseMatrix
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
from sympy.testing.pytest import raises
def test_symbolic_indexing():
    x, y, z, w = symbols('x y z w')
    M = ImmutableDenseNDimArray([[x, y], [z, w]])
    i, j = symbols('i, j')
    Mij = M[i, j]
    assert isinstance(Mij, Indexed)
    Ms = ImmutableSparseNDimArray([[2, 3 * x], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, Indexed)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = IndexedBase('A', (0, 2))
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]
    assert isinstance(M[3 * i - 2, j], Indexed)
    assert M[3 * i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], Indexed)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]
    assert M[i, j].diff(x) == ImmutableDenseNDimArray([[1, 0], [0, 0]])[i, j]
    assert Ms[i, j].diff(x) == ImmutableSparseNDimArray([[0, 3], [0, 0]])[i, j]
    Mo = ImmutableDenseNDimArray([1, 2, 3])
    assert Mo[i].subs(i, 1) == 2
    Mos = ImmutableSparseNDimArray([1, 2, 3])
    assert Mos[i].subs(i, 1) == 2
    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])
    raises(ValueError, lambda: Ms[i, 2])
    raises(ValueError, lambda: Ms[i, -1])
    raises(ValueError, lambda: Ms[2, i])
    raises(ValueError, lambda: Ms[-1, i])