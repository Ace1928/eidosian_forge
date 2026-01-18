import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_of_state_vector_as_mixture_pure_result_qudits():
    a = cirq.testing.random_superposition(2)
    b = cirq.testing.random_superposition(3)
    c = cirq.testing.random_superposition(4)
    state = np.kron(np.kron(a, b), c).reshape((2, 3, 4))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0], atol=1e-08), ((1.0, a),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-08), ((1.0, b),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [2], atol=1e-08), ((1.0, c),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-08), ((1.0, np.kron(a, b).reshape((2, 3))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [0, 2], atol=1e-08), ((1.0, np.kron(a, c).reshape((2, 4))),))
    assert mixtures_equal(cirq.partial_trace_of_state_vector_as_mixture(state, [1, 2], atol=1e-08), ((1.0, np.kron(b, c).reshape((3, 4))),))