from sympy.core.numbers import (Float, I, Integer)
from sympy.matrices.dense import Matrix
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.represent import (represent, rep_innerproduct,
from sympy.physics.quantum.state import Bra, Ket
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.matrixutils import (numpy_ndarray,
from sympy.physics.quantum.cartesian import XKet, XOp, XBra
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state
def test_scalar_sympy():
    assert represent(Integer(1)) == Integer(1)
    assert represent(Float(1.0)) == Float(1.0)
    assert represent(1.0 + I) == 1.0 + I