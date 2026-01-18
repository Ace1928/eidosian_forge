import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_not_power_of_two():
    with pytest.raises(ValueError, match='3'):
        cirq.sample_state_vector(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='5'):
        cirq.sample_state_vector(np.array([0, 1, 0, 0, 0]), [1])