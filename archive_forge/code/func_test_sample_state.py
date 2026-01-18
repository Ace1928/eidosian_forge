import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state():
    state = np.zeros(8, dtype=np.complex64)
    state[0] = 1 / np.sqrt(2)
    state[2] = 1 / np.sqrt(2)
    for _ in range(10):
        sample = cirq.sample_state_vector(state, [2, 1, 0])
        assert np.array_equal(sample, [[False, False, False]]) or np.array_equal(sample, [[False, True, False]])
    for _ in range(10):
        np.testing.assert_equal(cirq.sample_state_vector(state, [2]), [[False]])
        np.testing.assert_equal(cirq.sample_state_vector(state, [0]), [[False]])