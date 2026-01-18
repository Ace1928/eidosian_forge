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
def test_permutedims_with_indices():
    A = Array(range(32)).reshape(2, 2, 2, 2, 2)
    indices_new = list('abcde')
    indices_old = list('ebdac')
    new_A = permutedims(A, index_order_new=indices_new, index_order_old=indices_old)
    for a, b, c, d, e in itertools.product(range(2), range(2), range(2), range(2), range(2)):
        assert new_A[a, b, c, d, e] == A[e, b, d, a, c]
    indices_old = list('cabed')
    new_A = permutedims(A, index_order_new=indices_new, index_order_old=indices_old)
    for a, b, c, d, e in itertools.product(range(2), range(2), range(2), range(2), range(2)):
        assert new_A[a, b, c, d, e] == A[c, a, b, e, d]
    raises(ValueError, lambda: permutedims(A, index_order_old=list('aacde'), index_order_new=list('abcde')))
    raises(ValueError, lambda: permutedims(A, index_order_old=list('abcde'), index_order_new=list('abcce')))
    raises(ValueError, lambda: permutedims(A, index_order_old=list('abcde'), index_order_new=list('abce')))
    raises(ValueError, lambda: permutedims(A, index_order_old=list('abce'), index_order_new=list('abce')))
    raises(ValueError, lambda: permutedims(A, [2, 1, 0, 3, 4], index_order_old=list('abcde')))
    raises(ValueError, lambda: permutedims(A, [2, 1, 0, 3, 4], index_order_new=list('abcde')))