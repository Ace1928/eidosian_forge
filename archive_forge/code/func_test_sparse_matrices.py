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
def test_sparse_matrices():
    spm = SparseMatrix.diag(1, 0)
    assert unchanged(TensorProduct, spm, spm)