import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
def test_subspaces_size_3():
    plus_one_mod_3_gate = cirq.XPowGate(dimension=3)
    result = cirq.apply_unitary(unitary_value=plus_one_mod_3_gate, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(0, 1, 2)]))
    np.testing.assert_allclose(result, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=plus_one_mod_3_gate, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((3,), dtype=np.complex64), available_buffer=cirq.eye_tensor((3,), dtype=np.complex64), axes=(0,), subspaces=[(2, 1, 0)]))
    np.testing.assert_allclose(result, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), atol=1e-08)
    result = cirq.apply_unitary(unitary_value=plus_one_mod_3_gate, args=cirq.ApplyUnitaryArgs(target_tensor=cirq.eye_tensor((4,), dtype=np.complex64), available_buffer=cirq.eye_tensor((4,), dtype=np.complex64), axes=(0,), subspaces=[(1, 2, 3)]))
    np.testing.assert_allclose(result, np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]), atol=1e-08)