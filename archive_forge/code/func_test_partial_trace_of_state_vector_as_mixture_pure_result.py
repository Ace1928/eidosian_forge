import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_of_state_vector_as_mixture_pure_result():
    a = cirq.testing.random_superposition(4)
    b = cirq.testing.random_superposition(8)
    c = cirq.testing.random_superposition(16)
    state = np.kron(np.kron(a, b), c).reshape((2,) * 9)
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-08), ((1.0, a.reshape((2, 2))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4], atol=1e-08), ((1.0, b.reshape((2, 2, 2))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [5, 6, 7, 8], atol=1e-08), ((1.0, c.reshape((2, 2, 2, 2))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 2, 3, 4], atol=1e-08), ((1.0, np.kron(a, b).reshape((2, 2, 2, 2, 2))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 5, 6, 7, 8], atol=1e-08), ((1.0, np.kron(a, c).reshape((2, 2, 2, 2, 2, 2))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4, 5, 6, 7, 8], atol=1e-08), ((1.0, np.kron(b, c).reshape((2, 2, 2, 2, 2, 2, 2))),))
    state = state.reshape(2 ** 9)
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-08), ((1.0, a),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4], atol=1e-08), ((1.0, b),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [5, 6, 7, 8], atol=1e-08), ((1.0, c),))
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-20), truth, atol=1e-15)
    assert not mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-20), truth, atol=1e-16)