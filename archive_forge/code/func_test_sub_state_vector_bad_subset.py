import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_sub_state_vector_bad_subset():
    a = cirq.testing.random_superposition(4)
    b = cirq.testing.random_superposition(8)
    state = np.kron(a, b).reshape((2, 2, 2, 2, 2))
    for q1 in range(5):
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1e-08) is None
    for q1 in range(2):
        for q2 in range(2, 5):
            assert cirq.sub_state_vector(state, [q1, q2], default=None, atol=1e-08) is None
    for q3 in range(2, 5):
        assert cirq.sub_state_vector(state, [0, 1, q3], default=None, atol=1e-08) is None
    for q4 in range(2):
        assert cirq.sub_state_vector(state, [2, 3, 4, q4], default=None, atol=1e-08) is None