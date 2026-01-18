import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_quil_two_qubit_gate_eq():
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    gate2 = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    assert cirq.approx_eq(gate, gate2, atol=1e-08)
    gate3 = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    gate4 = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]))
    assert not cirq.approx_eq(gate4, gate3, atol=1e-08)