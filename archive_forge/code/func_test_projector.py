import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_projector(self, device, tol, skip_if):
    """Test that a tensor product involving qml.Projector works correctly"""
    n_wires = 3
    dev = device(n_wires)
    if isinstance(dev, qml.Device):
        if 'Projector' not in dev.observables:
            pytest.skip('Skipped because device does not support the Projector observable.')
        skip_if(dev, {'supports_tensor_observables': False})
    theta = 0.432
    phi = 0.123
    varphi = -0.543

    @qml.qnode(dev)
    def circuit(basis_state):
        qml.RX(theta, wires=[0])
        qml.RX(phi, wires=[1])
        qml.RX(varphi, wires=[2])
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.var(qml.Z(0) @ qml.Projector(basis_state, wires=[1, 2]))
    res_basis = circuit([0, 0])
    res_state = circuit([1, 0, 0, 0])
    expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([0, 1])
    res_state = circuit([0, 1, 0, 0])
    expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([1, 0])
    res_state = circuit([0, 0, 1, 0])
    expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([1, 1])
    res_state = circuit([0, 0, 0, 1])
    expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
    expected_mean = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2)
    expected_var = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2) - expected_mean ** 2
    assert np.allclose(res, expected_var, atol=tol(False))