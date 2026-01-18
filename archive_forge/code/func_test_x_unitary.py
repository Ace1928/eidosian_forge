import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_x_unitary():
    assert np.allclose(cirq.unitary(cirq.X), np.array([[0, 1], [1, 0]]))
    assert np.allclose(cirq.unitary(cirq.X ** 0.5), np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)
    assert np.allclose(cirq.unitary(cirq.X ** 0), np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.X ** (-0.5)), np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)