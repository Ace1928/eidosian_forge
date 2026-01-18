import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_var_projector(self, device, tol):
    """Tests if the samples of a Projector observable returned by sample have
        the correct values
        """
    n_wires = 2
    dev = device(n_wires)
    if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
        pytest.skip('Skipped because device does not support the Projector observable.')
    phi = 0.543
    theta = 0.654

    @qml.qnode(dev)
    def circuit(state):
        qml.RX(phi, wires=[0])
        qml.RY(theta, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.var(qml.Projector(state, wires=[0, 1]))
    res_basis = circuit([0, 0])
    res_state = circuit([1, 0, 0, 0])
    expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - ((np.cos(phi / 2) * np.cos(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([0, 1])
    res_state = circuit([0, 1, 0, 0])
    expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([1, 0])
    res_state = circuit([0, 0, 1, 0])
    expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res_basis = circuit([1, 1])
    res_state = circuit([0, 0, 0, 1])
    expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - ((np.sin(phi / 2) * np.cos(theta / 2)) ** 2) ** 2
    assert np.allclose(res_basis, expected, atol=tol(dev.shots))
    assert np.allclose(res_state, expected, atol=tol(dev.shots))
    res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
    expected_mean = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2)) ** 2)
    expected_var = expected_mean - expected_mean ** 2
    assert np.allclose(res, expected_var, atol=tol(dev.shots))