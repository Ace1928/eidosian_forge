import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_values_projector(self, device, tol):
    """Tests if the samples of a Projector observable returned by sample have
        the correct values
        """
    n_wires = 1
    dev = device(n_wires)
    if not dev.shots:
        pytest.skip('Device is in analytic mode, cannot test sampling.')
    if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
        pytest.skip('Skipped because device does not support the Projector observable.')
    theta = 0.543

    @qml.qnode(dev)
    def circuit(state):
        qml.RX(theta, wires=[0])
        return qml.sample(qml.Projector(state, wires=0))
    expected = np.cos(theta / 2) ** 2
    res_basis = circuit([0]).flatten()
    res_state = circuit([1, 0]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
    assert np.allclose(np.mean(res_state), expected, atol=tol(False))
    assert np.allclose(np.var(res_basis), expected - expected ** 2, atol=tol(False))
    assert np.allclose(np.var(res_state), expected - expected ** 2, atol=tol(False))
    expected = np.sin(theta / 2) ** 2
    res_basis = circuit([1]).flatten()
    res_state = circuit([0, 1]).flatten()
    assert np.allclose(sorted(list(set(res_basis.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(sorted(list(set(res_state.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res_basis), expected, atol=tol(False))
    assert np.allclose(np.mean(res_state), expected, atol=tol(False))
    assert np.allclose(np.var(res_basis), expected - expected ** 2, atol=tol(False))
    assert np.allclose(np.var(res_state), expected - expected ** 2, atol=tol(False))
    expected = 0.5
    res = circuit(np.array([1, 1]) / np.sqrt(2)).flatten()
    assert np.allclose(sorted(list(set(res.tolist()))), [0, 1], atol=tol(dev.shots))
    assert np.allclose(np.mean(res), expected, atol=tol(False))
    assert np.allclose(np.var(res), expected - expected ** 2, atol=tol(False))