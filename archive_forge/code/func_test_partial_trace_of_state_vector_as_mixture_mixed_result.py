import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_of_state_vector_as_mixture_mixed_result():
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    for q1 in [0, 1]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-08)
        assert mixtures_equal(mixture, truth)
    state = np.array([0, 1, 1, 0, 1, 0, 0, 0]).reshape((2, 2, 2)) / np.sqrt(3)
    truth = ((1 / 3, np.array([0.0, 1.0])), (2 / 3, np.array([1.0, 0.0])))
    for q1 in [0, 1, 2]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-08)
        assert mixtures_equal(mixture, truth)
    state = np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape((2, 2, 2)) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    for q1 in [0, 1, 2]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-08)
        assert mixtures_equal(mixture, truth)
    truth = ((0.5, np.array([1, 0, 0, 0]).reshape((2, 2))), (0.5, np.array([0, 0, 0, 1]).reshape((2, 2))))
    for q1, q2 in [(0, 1), (0, 2), (1, 2)]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1, q2], atol=1e-08)
        assert mixtures_equal(mixture, truth)