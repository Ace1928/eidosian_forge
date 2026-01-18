import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_rz_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.rz(np.pi / 2)), np.array([[s - s * 1j, 0], [0, s + s * 1j]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rz(-np.pi / 2)), np.array([[s + s * 1j, 0], [0, s - s * 1j]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rz(0)), np.array([[1, 0], [0, 1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rz(2 * np.pi)), np.array([[-1, 0], [0, -1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rz(np.pi)), np.array([[-1j, 0], [0, 1j]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rz(-np.pi)), np.array([[1j, 0], [0, -1j]]))