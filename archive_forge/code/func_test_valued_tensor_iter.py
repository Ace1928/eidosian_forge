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
def test_valued_tensor_iter():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        list_BA = [Array([1, 2, 3, 4]), Array([5, 6, 7, 8]), Array([9, 0, -1, -2]), Array([-3, -4, -5, -6])]
        assert list(A) == [E, px, py, pz]
        assert list(ba_matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6]
        assert list(BA) == list_BA
        assert list(A(i1)) == [E, px, py, pz]
        assert list(BA(i1, i2)) == list_BA
        assert list(3 * BA(i1, i2)) == [3 * i for i in list_BA]
        assert list(-5 * BA(i1, i2)) == [-5 * i for i in list_BA]
        assert list(A(i1) + A(i1)) == [2 * E, 2 * px, 2 * py, 2 * pz]
        assert BA(i1, i2) - BA(i1, i2) == 0
        assert list(BA(i1, i2) - 2 * BA(i1, i2)) == [-i for i in list_BA]