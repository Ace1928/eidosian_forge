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
def test_valued_tensor_contraction():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        assert (A(i0) * A(-i0)).data == E ** 2 - px ** 2 - py ** 2 - pz ** 2
        assert (A(i0) * A(-i0)).data == A ** 2
        assert (A(i0) * A(-i0)).data == A(i0) ** 2
        assert (A(i0) * B(-i0)).data == -px - 2 * py - 3 * pz
        for i in range(4):
            for j in range(4):
                assert (A(i0) * B(-i1))[i, j] == [E, px, py, pz][i] * [0, -1, -2, -3][j]
        assert (C(mu0) * C(-mu0)).data == -E ** 2 + px ** 2 + py ** 2 + pz ** 2
        contrexp = A(i0) * AB(i1, -i0)
        assert A(i0).rank == 1
        assert AB(i1, -i0).rank == 2
        assert contrexp.rank == 1
        for i in range(4):
            assert contrexp[i] == [E, px, py, pz][i]