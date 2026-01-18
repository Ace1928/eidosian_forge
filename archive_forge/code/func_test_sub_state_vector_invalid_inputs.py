import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_sub_state_vector_invalid_inputs():
    with pytest.raises(ValueError, match='7'):
        cirq.sub_state_vector(np.arange(7), [1, 2], atol=1e-08)
    with pytest.raises(ValueError, match='shaped'):
        cirq.sub_state_vector(np.arange(16).reshape((2, 4, 2)), [1, 2], atol=1e-08)
    with pytest.raises(ValueError, match='shaped'):
        cirq.sub_state_vector(np.arange(16).reshape((16, 1)), [1, 2], atol=1e-08)
    with pytest.raises(ValueError, match='normalized'):
        cirq.sub_state_vector(np.arange(16), [1, 2], atol=1e-08)
    state = np.arange(8) / np.linalg.norm(np.arange(8))
    with pytest.raises(ValueError, match='2, 2'):
        cirq.sub_state_vector(state, [1, 2, 2], atol=1e-08)
    state = np.array([1, 0, 0, 0]).reshape((2, 2))
    with pytest.raises(ValueError, match='invalid'):
        cirq.sub_state_vector(state, [5], atol=1e-08)
    with pytest.raises(ValueError, match='invalid'):
        cirq.sub_state_vector(state, [0, 1, 2], atol=1e-08)