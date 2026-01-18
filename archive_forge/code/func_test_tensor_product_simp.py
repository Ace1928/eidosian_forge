from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.core.expr import unchanged
from sympy.matrices import Matrix, SparseMatrix
from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import TensorProduct as TP
from sympy.physics.quantum.tensorproduct import tensor_product_simp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_tensor_product_simp():
    assert tensor_product_simp(TP(A, B) * TP(B, C)) == TP(A * B, B * C)
    assert tensor_product_simp(TP(A, B) ** x) == TP(A ** x, B ** x)
    assert tensor_product_simp(x * TP(A, B) ** 2) == x * TP(A ** 2, B ** 2)
    assert tensor_product_simp(x * TP(A, B) ** 2 * TP(C, D)) == x * TP(A ** 2 * C, B ** 2 * D)
    assert tensor_product_simp(TP(A, B) - TP(C, D) ** x) == TP(A, B) - TP(C ** x, D ** x)