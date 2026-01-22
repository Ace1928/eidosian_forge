import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@flaky(max_runs=10)
class TestTensorSample:
    """Test tensor sample values."""

    def test_paulix_pauliy(self, device, tol, skip_if):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
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
            return qml.sample(qml.X(0) @ qml.Y(2))
        res = circuit()
        assert np.allclose(res ** 2, 1, atol=tol(False))
        mean = np.mean(res)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2 - np.cos(2 * (theta - phi)) - np.cos(2 * (theta + phi)) + 2 * np.cos(2 * theta) + 2 * np.cos(2 * phi) + 14) / 16
        assert np.allclose(var, expected, atol=tol(False))

    def test_pauliz_hadamard(self, device, tol, skip_if):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
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
            return qml.sample(qml.Z(0) @ qml.Hadamard(wires=[1]) @ qml.Y(2))
        res = circuit()
        assert np.allclose(res ** 2, 1, atol=tol(False))
        mean = np.mean(res)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = (3 + np.cos(2 * phi) * np.cos(varphi) ** 2 - np.cos(2 * theta) * np.sin(varphi) ** 2 - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)) / 4
        assert np.allclose(var, expected, atol=tol(False))

    def test_hermitian(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device):
            if 'Hermitian' not in dev.observables:
                pytest.skip('Skipped because device does not support the Hermitian observable.')
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 0.432
        phi = 0.123
        varphi = -0.543
        A_ = 0.1 * np.array([[-6, 2 + 1j, -3, -5 + 2j], [2 - 1j, 0, 2 - 1j, -5 + 4j], [-3, 2 + 1j, 0, -4 + 3j], [-5 - 2j, -5 - 4j, -4 - 3j, -6]])

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.Z(0) @ qml.Hermitian(A_, wires=[1, 2]))
        res = circuit()
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A_))
        assert np.allclose(sorted(np.unique(res)), sorted(eigvals), atol=tol(False))
        mean = np.mean(res)
        expected = 0.1 * 0.5 * (-6 * np.cos(theta) * (np.cos(varphi) + 1) - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi)) + 3 * np.cos(varphi) * np.sin(phi) + np.sin(phi))
        assert np.allclose(mean, expected, atol=tol(False))
        var = np.var(res)
        expected = 0.01 * (1057 - np.cos(2 * phi) + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi) - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi)) + 16 * np.sin(2 * phi) - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi) - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2 - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi) - 8 * np.cos(theta) * (4 * np.cos(phi) * (4 + 8 * np.cos(varphi) + np.cos(2 * varphi) - (1 + 6 * np.cos(varphi)) * np.sin(varphi)) + np.sin(phi) * (15 + 8 * np.cos(varphi) - 11 * np.cos(2 * varphi) + 42 * np.sin(varphi) + 3 * np.sin(2 * varphi)))) / 16
        assert np.allclose(var, expected, atol=tol(False))

    def test_projector(self, device, tol, skip_if):
        """Test that a tensor product involving qml.Projector works correctly"""
        n_wires = 3
        dev = device(n_wires)
        if not dev.shots:
            pytest.skip('Device is in analytic mode, cannot test sampling.')
        if isinstance(dev, qml.Device):
            if 'Projector' not in dev.observables:
                pytest.skip('Skipped because device does not support the Projector observable.')
            skip_if(dev, {'supports_tensor_observables': False})
        theta = 1.432
        phi = 1.123
        varphi = -0.543

        @qml.qnode(dev)
        def circuit(state):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.sample(qml.Z(0) @ qml.Projector(state, wires=[1, 2]))
        res_basis = circuit([0, 0]).flatten()
        res_state = circuit([1, 0, 0, 0]).flatten()
        expected_mean = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        expected_var = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))
        res_basis = circuit([0, 1]).flatten()
        res_state = circuit([0, 1, 0, 0]).flatten()
        expected_mean = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        expected_var = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))
        res_basis = circuit([1, 0]).flatten()
        res_state = circuit([0, 0, 1, 0]).flatten()
        expected_mean = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        expected_var = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))
        res_basis = circuit([1, 1]).flatten()
        res_state = circuit([0, 0, 0, 1]).flatten()
        expected_mean = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        expected_var = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - ((np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2) ** 2
        assert np.allclose(sorted(np.unique(res_basis)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(sorted(np.unique(res_state)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res_basis), expected_mean, atol=tol(False))
        assert np.allclose(np.mean(res_state), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res_basis), expected_var, atol=tol(False))
        assert np.allclose(np.var(res_state), expected_var, atol=tol(False))
        res = circuit(np.array([1, 0, 0, 1]) / np.sqrt(2))
        expected_mean = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 - (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2)
        expected_var = 0.5 * ((np.cos(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.cos(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.sin(phi / 2) * np.cos(varphi / 2)) ** 2 + (np.sin(theta / 2) * np.cos(phi / 2) * np.cos(varphi / 2)) ** 2) - expected_mean ** 2
        assert np.allclose(sorted(np.unique(res)), [-1, 0, 1], atol=tol(False))
        assert np.allclose(np.mean(res), expected_mean, atol=tol(False))
        assert np.allclose(np.var(res), expected_var, atol=tol(False))