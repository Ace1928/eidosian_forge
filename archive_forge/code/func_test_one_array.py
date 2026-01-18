import random
from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
from sympy.testing.pytest import raises
def test_one_array():
    assert OneArray() == 1
    assert OneArray().is_Integer
    oa = OneArray(3, 2, 4)
    assert oa.shape == (3, 2, 4)
    oa_e = oa.as_explicit()
    assert oa_e.shape == (3, 2, 4)
    m, n, k = symbols('m n k')
    oa = OneArray(m, n, k, 2)
    assert oa.shape == (m, n, k, 2)
    raises(ValueError, lambda: oa.as_explicit())