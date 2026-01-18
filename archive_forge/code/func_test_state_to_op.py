from sympy.core.singleton import S
from sympy.physics.quantum.operatorset import (
from sympy.physics.quantum.cartesian import (
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises
def test_state_to_op():
    assert state_to_operators(XKet) == XOp()
    assert state_to_operators(PxKet) == PxOp()
    assert state_to_operators(XBra) == XOp()
    assert state_to_operators(PxBra) == PxOp()
    assert state_to_operators(Ket) == Operator()
    assert state_to_operators(Bra) == Operator()
    assert operators_to_state(state_to_operators(XKet('test'))) == XKet('test')
    assert operators_to_state(state_to_operators(XBra('test'))) == XKet('test')
    assert operators_to_state(state_to_operators(XKet())) == XKet()
    assert operators_to_state(state_to_operators(XBra())) == XKet()
    raises(NotImplementedError, lambda: state_to_operators(XOp))