from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_tensor_product_transpose():
    assert KroneckerProduct(I * A, B).transpose() == I * KroneckerProduct(A.transpose(), B.transpose())
    assert KroneckerProduct(mat1, mat2).transpose() == kronecker_product(mat1.transpose(), mat2.transpose())