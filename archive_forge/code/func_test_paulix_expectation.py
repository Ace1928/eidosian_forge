import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_paulix_expectation(self, device, tol):
    """Test that PauliX expectation value is correct"""
    n_wires = 2
    dev = device(n_wires)
    theta = 0.432
    phi = 0.123

    @qml.qnode(dev)
    def circuit():
        qml.RY(theta, wires=[0])
        qml.RY(phi, wires=[1])
        qml.CNOT(wires=[0, 1])
        return (qml.expval(qml.X(0)), qml.expval(qml.X(1)))
    res = circuit()
    expected = np.array([np.sin(theta) * np.sin(phi), np.sin(phi)])
    assert np.allclose(res, expected, atol=tol(dev.shots))