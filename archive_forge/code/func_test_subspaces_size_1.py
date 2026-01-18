import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_subspaces_size_1():
    phase_gate = cirq.MatrixGate(np.array([[1j]]))
    result = cirq.apply_unitary(unitary_value=phase_gate, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((2,), dtype=np.complex64), available_buffer=cirq.eye_tensor((2,), dtype=np.complex64), axes=(0,), subspaces=[(0,)]))
    np.testing.assert_allclose(result, np.array([[1j, 0], [0, 1]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=phase_gate, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((2,), dtype=np.complex64), available_buffer=cirq.eye_tensor((2,), dtype=np.complex64), axes=(0,), subspaces=[(1,)]))
    np.testing.assert_allclose(result, np.array([[1, 0], [0, 1j]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=phase_gate, args=cirq.ApplyUnitaryArgs(target_tensor=np.array([[0, 1], [1, 0]], dtype=np.complex64), available_buffer=np.zeros((2, 2), dtype=np.complex64), axes=(0,), subspaces=[(1,)]))
    np.testing.assert_allclose(result, np.array([[0, 1], [1j, 0]]), atol=1e-08)