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
def test_valued_metric_inverse():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        md = [[2, 2, 2, 1], [2, 3, 1, 0], [2, 1, 2, 3], [1, 0, 3, 2]]
        Lorentz.data = md
        m = Matrix(md)
        metric = Lorentz.metric
        minv = m.inv()
        meye = eye(4)
        KD = Lorentz.get_kronecker_delta()
        for i in range(4):
            for j in range(4):
                assert metric(i0, i1).data[i, j] == m[i, j]
                assert metric(-i0, -i1).data[i, j] == minv[i, j]
                assert metric(i0, -i1).data[i, j] == meye[i, j]
                assert metric(-i0, i1).data[i, j] == meye[i, j]
                assert metric(i0, i1)[i, j] == m[i, j]
                assert metric(-i0, -i1)[i, j] == minv[i, j]
                assert metric(i0, -i1)[i, j] == meye[i, j]
                assert metric(-i0, i1)[i, j] == meye[i, j]
                assert KD(i0, -i1)[i, j] == meye[i, j]