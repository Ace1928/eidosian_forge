import numpy as np
import cirq
import pytest
def test_state_prep_gate_printing():
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(np.array([1, 0, 0, 1]) / np.sqrt(2))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(gate(qubits[0], qubits[1]))
    cirq.testing.assert_has_diagram(circuit, '\n0: ───H───@───StatePreparation[1]───\n          │   │\n1: ───────X───StatePreparation[2]───\n')