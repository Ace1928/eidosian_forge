import numpy as np
import pytest
import cirq
import cirq.testing
def test_quantum_state_density_matrix():
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4
    state = cirq.quantum_state(density_matrix_1, qid_shape=(4,), copy=True)
    assert state.data is not density_matrix_1
    np.testing.assert_array_equal(state.data, density_matrix_1)
    assert state.qid_shape == (4,)
    assert state.dtype == np.complex64
    with pytest.raises(ValueError, match='not compatible'):
        _ = cirq.quantum_state(density_matrix_1, qid_shape=(8,))