import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_sample_state_negative_repetitions():
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(ValueError, match='-1'):
        cirq.sample_state_vector(state, [1], repetitions=-1)