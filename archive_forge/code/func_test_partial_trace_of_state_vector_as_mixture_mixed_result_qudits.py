import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_of_state_vector_as_mixture_mixed_result_qudits():
    state = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0, 0])), (0.5, np.array([0, 0, 1])))
    for q1 in [0, 1]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-08)
        assert mixtures_equal(mixture, truth)