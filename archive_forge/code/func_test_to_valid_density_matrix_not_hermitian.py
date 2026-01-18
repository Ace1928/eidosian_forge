import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_not_hermitian():
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[0.5, 0.5j], [0.5, 0.5j]]), num_qubits=1)
    with pytest.raises(ValueError, match='hermitian'):
        cirq.to_valid_density_matrix(np.array([[0.2, 0, 0, -0.2 - 0.3j], [0, 0, 0, 0], [0, 0, 0, 0], [0.2 + 0.3j, 0, 0, 0.8]]), num_qubits=2)