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
def test_tensor_product_commutator():
    assert TP(Comm(A, B), C).doit().expand(tensorproduct=True) == TP(A * B, C) - TP(B * A, C)
    assert Comm(TP(A, B), TP(B, C)).doit() == TP(A, B) * TP(B, C) - TP(B, C) * TP(A, B)