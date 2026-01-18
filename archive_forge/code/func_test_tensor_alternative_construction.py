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
def test_tensor_alternative_construction():
    L = TensorIndexType('L')
    i0, i1, i2, i3 = tensor_indices('i0:4', L)
    A = TensorHead('A', [L])
    x, y = symbols('x y')
    assert A(i0) == A(Symbol('i0'))
    assert A(-i0) == A(-Symbol('i0'))
    raises(TypeError, lambda: A(x + y))
    raises(ValueError, lambda: A(2 * x))