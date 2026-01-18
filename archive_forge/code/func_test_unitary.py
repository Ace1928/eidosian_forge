from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye
def test_unitary():
    U = UnitaryOperator('U')
    assert isinstance(U, UnitaryOperator)
    assert isinstance(U, Operator)
    assert U.inv() == Dagger(U)
    assert U * Dagger(U) == 1
    assert Dagger(U) * U == 1
    assert U.is_commutative is False
    assert Dagger(U).is_commutative is False