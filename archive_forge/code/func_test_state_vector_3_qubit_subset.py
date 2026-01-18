from math import sqrt, pi
import pytest
import numpy as np
from flaky import flaky
import pennylane as qml
@pytest.mark.parametrize('par,wires,expected_output', [([1j / np.sqrt(10), (1 - 2j) / np.sqrt(10), 0, 0, 0, 2 / np.sqrt(10), 0, 0], [0, 1, 2], [1 / 5.0, 1.0, -4 / 5.0]), ([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 2], [0.0, 1.0, 0.0]), ([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 1], [0.0, 0.0, 1.0]), ([0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 0], [-1.0, 1.0, 1.0]), ([0, 1j, 0, 0, 0, 0, 0, 0], [0, 2, 1], [1.0, -1.0, 1.0]), ([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0], [-1.0, 0.0, 1.0]), ([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1], [0.0, -1.0, 1.0])])
def test_state_vector_3_qubit_subset(self, device, tol, par, wires, expected_output):
    """Tests qubit state vector preparation on subsets of 3 qubits"""
    n_wires = 3
    dev = device(n_wires)
    par = np.array(par)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(par, wires=wires)
        return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2)))
    assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))