import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_density_matrix_wrong_dtype():
    with pytest.raises(ValueError, match='dtype'):
        cirq.to_valid_density_matrix(np.array([[1, 0], [0, 0]], dtype=np.complex64), num_qubits=1, dtype=np.complex128)