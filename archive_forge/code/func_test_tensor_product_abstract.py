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
def test_tensor_product_abstract():
    assert TP(x * A, 2 * B) == x * 2 * TP(A, B)
    assert TP(A, B) != TP(B, A)
    assert TP(A, B).is_commutative is False
    assert isinstance(TP(A, B), TP)
    assert TP(A, B).subs(A, C) == TP(C, B)