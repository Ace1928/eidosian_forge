import random
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('max_ms_depth,effect', [(0, np.eye(4)), (0, np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0j]])), (1, cirq.unitary(cirq.ms(np.pi / 4))), (0, cirq.unitary(cirq.CZ ** 1e-08)), (0.5, cirq.unitary(cirq.CZ ** 0.5)), (1, cirq.unitary(cirq.CZ)), (1, cirq.unitary(cirq.CNOT)), (1, np.array([[1, 0, 0, 1j], [0, 1, 1j, 0], [0, 1j, 1, 0], [1j, 0, 0, 1]]) * np.sqrt(0.5)), (1, np.array([[1, 0, 0, -1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [-1j, 0, 0, 1]]) * np.sqrt(0.5)), (1, np.array([[1, 0, 0, 1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [1j, 0, 0, 1]]) * np.sqrt(0.5)), (1.5, cirq.map_eigenvalues(cirq.unitary(cirq.SWAP), lambda e: e ** 0.5)), (2, cirq.unitary(cirq.SWAP).dot(cirq.unitary(cirq.CZ))), (3, cirq.unitary(cirq.SWAP)), (3, np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0j]]))] + [(1, _random_single_MS_effect()) for _ in range(10)] + [(3, cirq.testing.random_unitary(4)) for _ in range(10)] + [(2, _random_double_MS_effect()) for _ in range(10)])
def test_two_to_ops(max_ms_depth: int, effect: np.ndarray):
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    operations = cirq.two_qubit_matrix_to_ion_operations(q0, q1, effect)
    assert_ops_implement_unitary(q0, q1, operations, effect)
    assert_ms_depth_below(operations, max_ms_depth)