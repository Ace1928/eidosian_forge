import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_rx_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.rx(np.pi / 2)), np.array([[s, -s * 1j], [-s * 1j, s]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rx(-np.pi / 2)), np.array([[s, s * 1j], [s * 1j, s]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rx(0)), np.array([[1, 0], [0, 1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rx(2 * np.pi)), np.array([[-1, 0], [0, -1]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rx(np.pi)), np.array([[0, -1j], [-1j, 0]]))
    np.testing.assert_allclose(cirq.unitary(cirq.rx(-np.pi)), np.array([[0, 1j], [1j, 0]]))