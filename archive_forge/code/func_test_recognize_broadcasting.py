from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
from sympy.testing.pytest import raises
def test_recognize_broadcasting():
    expr = ArrayTensorProduct(x.T * x, A)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(x.T * x, A), [0, 1])
    expr = ArrayTensorProduct(A, x.T * x)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(A, x.T * x), [2, 3])
    expr = ArrayTensorProduct(A, B, x.T * x, C)
    assert _remove_trivial_dims(expr) == (ArrayTensorProduct(A, KroneckerProduct(B, x.T * x), C), [4, 5])
    expr = ArrayTensorProduct(a, b, x.T * x)
    assert _remove_trivial_dims(expr) == (a * x.T * x * b.T, [1, 3, 4, 5])