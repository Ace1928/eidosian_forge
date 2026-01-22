import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@flaky(max_runs=10)
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, tol):
        """Test that identity expectation value (i.e. the trace) is 1."""
        n_wires = 2
        dev = device(n_wires)
        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return (qml.expval(qml.Identity(wires=0)), qml.expval(qml.Identity(wires=1)))
        res = circuit()
        assert np.allclose(res, np.array([1, 1]), atol=tol(dev.shots))

    def test_pauliz_expectation(self, device, tol):
        """Test that PauliZ expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return (qml.expval(qml.Z(0)), qml.expval(qml.Z(1)))
        res = circuit()
        expected = np.array([np.cos(theta), np.cos(theta) * np.cos(phi)])
        assert np.allclose(res, expected, atol=tol(dev.shots))

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

    def test_hadamard_expectation(self, device, tol):
        """Test that Hadamard expectation value is correct"""
        n_wires = 2
        dev = device(n_wires)
        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return (qml.expval(qml.Hadamard(wires=0)), qml.expval(qml.Hadamard(wires=1)))
        res = circuit()
        expected = np.array([np.sin(theta) * np.sin(phi) + np.cos(theta), np.cos(theta) * np.cos(phi) + np.sin(phi)]) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_hermitian_expectation(self, device, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
            pytest.skip('Skipped because device does not support the Hermitian observable.')
        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return (qml.expval(qml.Hermitian(A, wires=0)), qml.expval(qml.Hermitian(A, wires=1)))
        res = circuit()
        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_projector_expectation(self, device, tol):
        """Test that arbitrary Projector expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qml.Device) and 'Projector' not in dev.observables:
            pytest.skip('Skipped because device does not support the Projector observable.')
        theta = 0.732
        phi = 0.523

        @qml.qnode(dev)
        def circuit(state):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector(state, wires=[0, 1]))
        basis_state, state_vector = ([0, 0], [1, 0, 0, 0])
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))
        basis_state, state_vector = ([0, 1], [0, 1, 0, 0])
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))
        basis_state, state_vector = ([1, 0], [0, 0, 1, 0])
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))
        basis_state, state_vector = ([1, 1], [0, 0, 0, 1])
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(circuit(basis_state), expected, atol=tol(dev.shots))
        assert np.allclose(circuit(state_vector), expected, atol=tol(dev.shots))

    def test_multi_mode_hermitian_expectation(self, device, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        n_wires = 2
        dev = device(n_wires)
        if isinstance(dev, qml.Device) and 'Hermitian' not in dev.observables:
            pytest.skip('Skipped because device does not support the Hermitian observable.')
        theta = 0.432
        phi = 0.123
        A_ = np.array([[-6, 2 + 1j, -3, -5 + 2j], [2 - 1j, 0, 2 - 1j, -5 + 4j], [-3, 2 + 1j, 0, -4 + 3j], [-5 - 2j, -5 - 4j, -4 - 3j, -6]])

        @qml.qnode(dev)
        def circuit():
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A_, wires=[0, 1]))
        res = circuit()
        expected = 0.5 * (6 * np.cos(theta) * np.sin(phi) - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3) - 2 * np.sin(phi) - 6 * np.cos(phi) - 6)
        assert np.allclose(res, expected, atol=tol(dev.shots))