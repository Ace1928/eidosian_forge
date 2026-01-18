import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_wrong_dims():
    x3 = cirq.XPowGate(dimension=3)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = x3.on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = x3.on(cirq.LineQid(0, dimension=4))
    z3 = cirq.ZPowGate(dimension=3)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = z3.on(cirq.LineQubit(0))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = z3.on(cirq.LineQid(0, dimension=4))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = cirq.X.on(cirq.LineQid(0, dimension=3))
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = cirq.Z.on(cirq.LineQid(0, dimension=3))