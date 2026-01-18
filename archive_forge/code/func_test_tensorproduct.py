import itertools
import random
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.testing.pytest import raises
from sympy.core.function import diff
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.array import Array, ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray
from sympy.tensor.array.arrayop import tensorproduct, tensorcontraction, derive_by_array, permutedims, Flatten, \
def test_tensorproduct():
    x, y, z, t = symbols('x y z t')
    from sympy.abc import a, b, c, d
    assert tensorproduct() == 1
    assert tensorproduct([x]) == Array([x])
    assert tensorproduct([x], [y]) == Array([[x * y]])
    assert tensorproduct([x], [y], [z]) == Array([[[x * y * z]]])
    assert tensorproduct([x], [y], [z], [t]) == Array([[[[x * y * z * t]]]])
    assert tensorproduct(x) == x
    assert tensorproduct(x, y) == x * y
    assert tensorproduct(x, y, z) == x * y * z
    assert tensorproduct(x, y, z, t) == x * y * z * t
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        A = ArrayType([x, y])
        B = ArrayType([1, 2, 3])
        C = ArrayType([a, b, c, d])
        assert tensorproduct(A, B, C) == ArrayType([[[a * x, b * x, c * x, d * x], [2 * a * x, 2 * b * x, 2 * c * x, 2 * d * x], [3 * a * x, 3 * b * x, 3 * c * x, 3 * d * x]], [[a * y, b * y, c * y, d * y], [2 * a * y, 2 * b * y, 2 * c * y, 2 * d * y], [3 * a * y, 3 * b * y, 3 * c * y, 3 * d * y]]])
        assert tensorproduct([x, y], [1, 2, 3]) == tensorproduct(A, B)
        assert tensorproduct(A, 2) == ArrayType([2 * x, 2 * y])
        assert tensorproduct(A, [2]) == ArrayType([[2 * x], [2 * y]])
        assert tensorproduct([2], A) == ArrayType([[2 * x, 2 * y]])
        assert tensorproduct(a, A) == ArrayType([a * x, a * y])
        assert tensorproduct(a, A, B) == ArrayType([[a * x, 2 * a * x, 3 * a * x], [a * y, 2 * a * y, 3 * a * y]])
        assert tensorproduct(A, B, a) == ArrayType([[a * x, 2 * a * x, 3 * a * x], [a * y, 2 * a * y, 3 * a * y]])
        assert tensorproduct(B, a, A) == ArrayType([[a * x, a * y], [2 * a * x, 2 * a * y], [3 * a * x, 3 * a * y]])
    for SparseArrayType in [ImmutableSparseNDimArray, MutableSparseNDimArray]:
        a = SparseArrayType({1: 2, 3: 4}, (1000, 2000))
        b = SparseArrayType({1: 2, 3: 4}, (1000, 2000))
        assert tensorproduct(a, b) == ImmutableSparseNDimArray({2000001: 4, 2000003: 8, 6000001: 8, 6000003: 16}, (1000, 2000, 1000, 2000))