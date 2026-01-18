import numpy as np
import cirq
import pytest
@pytest.mark.parametrize('name', ['Prep', 'S'])
def test_state_prep_gate_printing_with_name(name):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(np.array([1, 0, 0, 1]) / np.sqrt(2), name=name)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(gate(qubits[0], qubits[1]))
    cirq.testing.assert_has_diagram(circuit, f'\n0: ───H───@───{name}[1]───\n          │   │\n1: ───────X───{name}[2]───\n')