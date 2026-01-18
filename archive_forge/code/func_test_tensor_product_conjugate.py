from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace
from sympy.matrices.expressions.kronecker import (KroneckerProduct,
def test_tensor_product_conjugate():
    assert KroneckerProduct(I * A, B).conjugate() == -I * KroneckerProduct(A.conjugate(), B.conjugate())
    assert KroneckerProduct(mat1, mat2).conjugate() == kronecker_product(mat1.conjugate(), mat2.conjugate())