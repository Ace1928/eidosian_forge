from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_valued_assign_numpy_ndarray():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        arr = [E + 1, px - 1, py, pz]
        A.data = Array(arr)
        for i in range(4):
            assert A(i0).data[i] == arr[i]
        qx, qy, qz = symbols('qx qy qz')
        A(-i0).data = Array([E, qx, qy, qz])
        for i in range(4):
            assert A(i0).data[i] == [E, -qx, -qy, -qz][i]
            assert A.data[i] == [E, -qx, -qy, -qz][i]
        random_4x4_data = [[(i ** 3 - 3 * i ** 2) % (j + 7) for i in range(4)] for j in range(4)]
        AB(-i0, -i1).data = random_4x4_data
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j] * (-1 if i else 1) * (-1 if j else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j] * (-1 if j else 1)
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j] * (-1 if i else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]
        AB(-i0, i1).data = random_4x4_data
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j] * (-1 if i else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j] * (-1 if i else 1) * (-1 if j else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j] * (-1 if j else 1)