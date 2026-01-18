import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_from_state_vector():
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(density_matrix_rep=np.array([1, 0], dtype=np.complex64), num_qubits=1), np.array([[1, 0], [0, 0]]))
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(density_matrix_rep=np.array([np.sqrt(0.3), np.sqrt(0.7)], dtype=np.complex64), num_qubits=1), np.array([[0.3, np.sqrt(0.3 * 0.7)], [np.sqrt(0.3 * 0.7), 0.7]]))
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(density_matrix_rep=np.array([np.sqrt(0.5), np.sqrt(0.5) * 1j], dtype=np.complex64), num_qubits=1), np.array([[0.5, -0.5j], [0.5j, 0.5]]))
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(density_matrix_rep=np.array([0.5] * 4, dtype=np.complex64), num_qubits=2), 0.25 * np.ones((4, 4)))