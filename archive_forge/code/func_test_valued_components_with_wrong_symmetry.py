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
def test_valued_components_with_wrong_symmetry():
    with warns_deprecated_sympy():
        IT = TensorIndexType('IT', dim=3)
        i0, i1, i2, i3 = tensor_indices('i0:4', IT)
        IT.data = [1, 1, 1]
        A_nosym = TensorHead('A', [IT] * 2)
        A_sym = TensorHead('A', [IT] * 2, TensorSymmetry.fully_symmetric(2))
        A_antisym = TensorHead('A', [IT] * 2, TensorSymmetry.fully_symmetric(-2))
        mat_nosym = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mat_sym = mat_nosym + mat_nosym.T
        mat_antisym = mat_nosym - mat_nosym.T
        A_nosym.data = mat_nosym
        A_nosym.data = mat_sym
        A_nosym.data = mat_antisym

        def assign(A, dat):
            A.data = dat
        A_sym.data = mat_sym
        raises(ValueError, lambda: assign(A_sym, mat_nosym))
        raises(ValueError, lambda: assign(A_sym, mat_antisym))
        A_antisym.data = mat_antisym
        raises(ValueError, lambda: assign(A_antisym, mat_sym))
        raises(ValueError, lambda: assign(A_antisym, mat_nosym))
        A_sym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        A_antisym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]