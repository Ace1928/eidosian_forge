import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_big_endian():
    results = []
    for x in range(8):
        state = cirq.to_valid_state_vector(x, 3)
        sample = cirq.sample_state_vector(state, [2, 1, 0])
        results.append(sample)
    expecteds = [[list(reversed(x))] for x in list(itertools.product([False, True], repeat=3))]
    for result, expected in zip(results, expecteds):
        np.testing.assert_equal(result, expected)