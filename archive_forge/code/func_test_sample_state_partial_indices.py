import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_partial_indices():
    for index in range(3):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            np.testing.assert_equal(cirq.sample_state_vector(state, [index]), [[bool(1 & x >> 2 - index)]])