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
def test_valued_tensor_add_scalar():
    with warns_deprecated_sympy():
        A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4 = _get_valued_base_test_variables()
        expr1 = A(i0) * A(-i0) - (E ** 2 - px ** 2 - py ** 2 - pz ** 2)
        assert expr1.data == 0
        expr2 = E ** 2 - px ** 2 - py ** 2 - pz ** 2 - A(i0) * A(-i0)
        assert expr2.data == 0
        expr3 = A(i0) * A(-i0) - E ** 2 + px ** 2 + py ** 2 + pz ** 2
        assert expr3.data == 0
        expr4 = C(mu0) * C(-mu0) + 2 * E ** 2 - 2 * px ** 2 - 2 * py ** 2 - 2 * pz ** 2 - A(i0) * A(-i0)
        assert expr4.data == 0