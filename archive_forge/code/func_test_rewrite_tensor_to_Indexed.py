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
def test_rewrite_tensor_to_Indexed():
    L = TensorIndexType('L', dim=4)
    A = TensorHead('A', [L] * 4)
    B = TensorHead('B', [L])
    i0, i1, i2, i3 = symbols('i0:4')
    L_0, L_1 = symbols('L_0:2')
    a1 = A(i0, i1, i2, i3)
    assert a1.rewrite(Indexed) == Indexed(Symbol('A'), i0, i1, i2, i3)
    a2 = A(i0, -i0, i2, i3)
    assert a2.rewrite(Indexed) == Sum(Indexed(Symbol('A'), L_0, L_0, i2, i3), (L_0, 0, 3))
    a3 = a2 + A(i2, i3, i0, -i0)
    assert a3.rewrite(Indexed) == Sum(Indexed(Symbol('A'), L_0, L_0, i2, i3), (L_0, 0, 3)) + Sum(Indexed(Symbol('A'), i2, i3, L_0, L_0), (L_0, 0, 3))
    b1 = B(-i0) * a1
    assert b1.rewrite(Indexed) == Sum(Indexed(Symbol('B'), L_0) * Indexed(Symbol('A'), L_0, i1, i2, i3), (L_0, 0, 3))
    b2 = B(-i3) * a2
    assert b2.rewrite(Indexed) == Sum(Indexed(Symbol('B'), L_1) * Indexed(Symbol('A'), L_0, L_0, i2, L_1), (L_0, 0, 3), (L_1, 0, 3))