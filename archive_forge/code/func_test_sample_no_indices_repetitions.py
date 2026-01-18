import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_no_indices_repetitions():
    state = cirq.to_valid_state_vector(0, 3)
    np.testing.assert_almost_equal(cirq.sample_state_vector(state, [], repetitions=2), np.zeros(shape=(2, 0)))