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
def test_valued_canon_bp_swapaxes():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        e1 = A(i1) * A(i0)
        e2 = e1.canon_bp()
        assert e2 == A(i0) * A(i1)
        for i in range(4):
            for j in range(4):
                assert e1[i, j] == e2[j, i]
        o1 = B(i2) * A(i1) * B(i0)
        o2 = o1.canon_bp()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    assert o1[i, j, k] == o2[j, i, k]