import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_empty_state():
    state = np.array([1.0])
    np.testing.assert_almost_equal(cirq.sample_state_vector(state, []), np.zeros(shape=(1, 0)))