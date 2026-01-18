from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.external import import_module
from sympy.physics.quantum.density import Density, entropy, fidelity
from sympy.physics.quantum.state import Ket, TimeDepKet
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.cartesian import XKet, PxKet, PxOp, XOp
from sympy.physics.quantum.spin import JzKet
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.trace import Tr
from sympy.functions import sqrt
from sympy.testing.pytest import raises
from sympy.physics.quantum.matrixutils import scipy_sparse_matrix
from sympy.physics.quantum.tensorproduct import TensorProduct
def test_probs():
    d = Density([Ket(0), 0.75], [Ket(1), 0.25])
    probs = d.probs()
    assert probs[0] == 0.75 and probs[1] == 0.25
    x, y = symbols('x y')
    d = Density([Ket(0), x], [Ket(1), y])
    probs = d.probs()
    assert probs[0] == x and probs[1] == y