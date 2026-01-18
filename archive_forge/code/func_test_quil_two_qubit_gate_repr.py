import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_quil_two_qubit_gate_repr():
    gate = cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    assert repr(gate) == 'cirq.circuits.quil_output.QuilTwoQubitGate(matrix=\n[[1 0 0 0]\n [0 1 0 0]\n [0 0 1 0]\n [0 0 0 1]]\n)'