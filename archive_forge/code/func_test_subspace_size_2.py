import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_subspace_size_2():
    result = cirq.apply_unitary(unitary_value=cirq.X, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(0, 1)]))
    np.testing.assert_allclose(result, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=cirq.X, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(0, 2)]))
    np.testing.assert_allclose(result, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=cirq.X, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(1, 2)]))
    np.testing.assert_allclose(result, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=cirq.X, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((4,), dtype=np.complex64), available_buffer=cirq.eye_tensor((4,), dtype=np.complex64), axes=(0,), subspaces=[(1, 2)]))
    np.testing.assert_allclose(result, np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), atol=1e-08)