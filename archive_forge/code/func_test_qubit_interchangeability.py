import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('gate, interchangeable', ((cirq.PhasedFSimGate(1, 2, 3, 4, 5), False), (cirq.PhasedFSimGate(1, 2, 0, 4, 5), False), (cirq.PhasedFSimGate(1, 0, 3, 4, 5), False), (cirq.PhasedFSimGate(1, 0, 0, 4, 5), True), (cirq.PhasedFSimGate(np.pi / 2, 2, 0, 4, 5), True), (cirq.PhasedFSimGate(np.pi, 0, 3, 4, 5), True), (cirq.PhasedFSimGate(1, -np.pi, 0, 4, 5), True), (cirq.PhasedFSimGate(1, 0, np.pi, 4, 5), True), (cirq.PhasedFSimGate(1, np.pi / 2, 0, 4, 5), False)))
def test_qubit_interchangeability(gate, interchangeable):
    a, b = cirq.LineQubit.range(2)
    c1 = cirq.Circuit(gate.on(a, b))
    c2 = cirq.Circuit(cirq.SWAP(a, b), gate.on(a, b), cirq.SWAP(a, b))
    u1 = cirq.unitary(c1)
    u2 = cirq.unitary(c2)
    assert np.all(u1 == u2) == interchangeable