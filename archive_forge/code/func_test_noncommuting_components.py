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
def test_noncommuting_components():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        euclid = TensorIndexType('Euclidean')
        euclid.data = [1, 1]
        i1, i2, i3 = tensor_indices('i1:4', euclid)
        a, b, c, d = symbols('a b c d', commutative=False)
        V1 = TensorHead('V1', [euclid] * 2)
        V1.data = [[a, b], (c, d)]
        V2 = TensorHead('V2', [euclid] * 2)
        V2.data = [[a, c], [b, d]]
        vtp = V1(i1, i2) * V2(-i2, -i1)
        assert vtp.data == a ** 2 + b ** 2 + c ** 2 + d ** 2
        assert vtp.data != a ** 2 + 2 * b * c + d ** 2
        vtp2 = V1(i1, i2) * V1(-i2, -i1)
        assert vtp2.data == a ** 2 + b * c + c * b + d ** 2
        assert vtp2.data != a ** 2 + 2 * b * c + d ** 2
        Vc = (b * V1(i1, -i1)).data
        assert Vc.expand() == b * a + b * d