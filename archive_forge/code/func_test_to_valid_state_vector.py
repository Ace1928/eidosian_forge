import numpy as np
import pytest
import cirq
import cirq.testing
def test_to_valid_state_vector():
    with pytest.raises(ValueError, match='Computational basis state is out of range'):
        cirq.to_valid_state_vector(2, 1)
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64), 2), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64), 2), np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(0, 2), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(1, 2), np.array([0.0, 1.0, 0.0, 0.0]))
    v = cirq.to_valid_state_vector([0, 1, 2, 0], qid_shape=(3, 3, 3, 3))
    assert v.shape == (3 ** 4,)
    assert v[6 + 9] == 1
    v = cirq.to_valid_state_vector([False, True, False, False], num_qubits=4)
    assert v.shape == (16,)
    assert v[4] == 1
    v = cirq.to_valid_state_vector([0, 1, 0, 0], num_qubits=2)
    assert v.shape == (4,)
    assert v[1] == 1
    v = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.complex64), qid_shape=(2, 1))
    assert v.shape == (2,)
    assert v[0] == 1