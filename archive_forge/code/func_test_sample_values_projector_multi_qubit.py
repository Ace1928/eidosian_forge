import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_values_projector_multi_qubit(self, device, tol):
    """Tests if the samples of a multi-qubit Projector observable returned by sample have
        the correct values
        """
    n_wires = 2
    dev = device(n_wires)
    if not dev.shots:
        pytest.skip('Device is in analytic mode, cannot test sampling.')
    if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
        pytest.skip('Skipped because device does not support the Projector observable.')
    theta = 0.543

    @qml.qnode(dev)
    def circuit(state):
        qml.RX(theta, wires=[0])
        qml.RY(2 * theta, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.Projector(state, wires=[0, 1]))
    expected = (np.cos(theta / 2) * np.cos(theta)) ** 2
    res_basis = circuit([0, 0]).flatten()
    res_state = circuit([1, 0, 0, 0]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
    assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
    expected = (np.cos(theta / 2) * np.sin(theta)) ** 2
    res_basis = circuit([0, 1]).flatten()
    res_state = circuit([0, 1, 0, 0]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
    assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
    expected = (np.sin(theta / 2) * np.sin(theta)) ** 2
    res_basis = circuit([1, 0]).flatten()
    res_state = circuit([0, 0, 1, 0]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
    assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))
    expected = (np.sin(theta / 2) * np.cos(theta)) ** 2
    res_basis = circuit([1, 1]).flatten()
    res_state = circuit([0, 0, 0, 1]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(dev.shots))
    assert np.allclose(np.mean(res_state), expected, atol=tol(dev.shots))