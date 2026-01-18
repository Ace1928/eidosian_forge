import math
import cirq
import pytest
import numpy as np
import cirq_web
def test_valid_bloch_sphere_vector():
    state_vector = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
    bloch_sphere = cirq_web.BlochSphere(state_vector=state_vector)
    bloch_vector = cirq.bloch_vector_from_state_vector(state_vector, 0)
    assert np.array_equal(bloch_vector, bloch_sphere.bloch_vector)