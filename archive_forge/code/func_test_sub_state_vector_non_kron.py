import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_sub_state_vector_non_kron():
    a = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
    b = np.array([1, 1]) / np.sqrt(2)
    state = np.kron(a, b).reshape((2, 2, 2, 2))
    for q1 in [0, 1, 2]:
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1e-08) is None
    for q1 in [0, 1, 2]:
        assert cirq.sub_state_vector(state, [q1, 3], default=None, atol=1e-08) is None
    with pytest.raises(ValueError, match='factored'):
        _ = cirq.sub_state_vector(a, [0], atol=1e-08)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(state, [3]), b, atol=1e-08)