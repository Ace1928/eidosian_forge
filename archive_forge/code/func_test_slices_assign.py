from copy import copy
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices import SparseMatrix
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import MutableSparseNDimArray
from sympy.testing.pytest import raises
def test_slices_assign():
    a = MutableDenseNDimArray(range(12), shape=(4, 3))
    b = MutableSparseNDimArray(range(12), shape=(4, 3))
    for i in [a, b]:
        assert i.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, :] = [2, 2, 2]
        assert i.tolist() == [[2, 2, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[0, 1:] = [8, 8]
        assert i.tolist() == [[2, 8, 8], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        i[1:3, 1] = [20, 44]
        assert i.tolist() == [[2, 8, 8], [3, 20, 5], [6, 44, 8], [9, 10, 11]]