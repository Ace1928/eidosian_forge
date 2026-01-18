import numpy as np
import pytest
import cirq
import cirq.testing
def test_validate_normalized_state():
    cirq.validate_normalized_state_vector(cirq.testing.random_superposition(2), qid_shape=(2,))
    cirq.validate_normalized_state_vector(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64), qid_shape=(2, 2))
    with pytest.raises(ValueError, match='invalid dtype'):
        cirq.validate_normalized_state_vector(np.array([1, 1], dtype=np.complex64), qid_shape=(2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match='incorrect size'):
        cirq.validate_normalized_state_vector(np.array([1, 1], dtype=np.complex64), qid_shape=(2, 2))
    with pytest.raises(ValueError, match='not normalized'):
        cirq.validate_normalized_state_vector(np.array([1.0, 0.2, 0.0, 0.0], dtype=np.complex64), qid_shape=(2, 2))