import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def test_sample_values_hermitian_multi_qubit(self, device, tol):
    """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
    n_wires = 2
    dev = device(n_wires)
    if not dev.shots:
        pytest.skip('Device is in analytic mode, cannot test sampling.')
    if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
        pytest.skip('Skipped because device does not support the Hermitian observable.')
    theta = 0.543
    A_ = np.array([[1, 2j, 1 - 2j, 0.5j], [-2j, 0, 3 + 4j, 1], [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j], [-0.5j, 1, 1.5 + 2j, -1]])

    @qml.qnode(dev)
    def circuit():
        qml.RX(theta, wires=[0])
        qml.RY(2 * theta, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.Hermitian(A_, wires=[0, 1]))
    res = circuit().flatten()
    eigvals = np.linalg.eigvalsh(A_)
    assert np.allclose(sorted(list(set(res.tolist()))), sorted(eigvals), atol=tol(dev.shots))
    expected = (88 * np.sin(theta) + 24 * np.sin(2 * theta) - 40 * np.sin(3 * theta) + 5 * np.cos(theta) - 6 * np.cos(2 * theta) + 27 * np.cos(3 * theta) + 6) / 32
    assert np.allclose(np.mean(res), expected, atol=tol(dev.shots))