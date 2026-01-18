import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_pauliy_expectation(self, device, tol):
    """Test that PauliY expectation value is correct"""
    n_wires = 2
    dev = device(n_wires)
    theta = 0.432
    phi = 0.123

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return (qml.expval(qml.Y(0)), qml.expval(qml.Y(1)))
    res = circuit()
    expected = np.array([0.0, -np.cos(theta) * np.sin(phi)])
    assert np.allclose(res, expected, atol=tol(dev.shots))