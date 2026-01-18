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
def test_tensorhead_construction_without_symmetry():
    L = TensorIndexType('Lorentz')
    A1 = TensorHead('A', [L, L])
    A2 = TensorHead('A', [L, L], TensorSymmetry.no_symmetry(2))
    assert A1 == A2
    A3 = TensorHead('A', [L, L], TensorSymmetry.fully_symmetric(2))
    assert A1 != A3