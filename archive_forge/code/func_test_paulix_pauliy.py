import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_paulix_pauliy(self, device, tol, skip_if):
    """Test that a tensor product involving PauliX and PauliY works correctly"""
    n_wires = 3
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'supports_tensor_observables': False})
    theta = 0.432
    phi = 0.123
    varphi = -0.543

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.RX(varphi, wires=[2])
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.var(qml.X(0) @ qml.Y(2))
    res = circuit()
    expected = (8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2 - np.cos(2 * (theta - phi)) - np.cos(2 * (theta + phi)) + 2 * np.cos(2 * theta) + 2 * np.cos(2 * phi) + 14) / 16
    assert np.allclose(res, expected, atol=tol(dev.shots))