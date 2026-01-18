import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def test_random_qubit_default():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary(randomize_global_phase=True, rng=rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((1, 1, 1), True, rng=rng).ravel()
    np.testing.assert_almost_equal(actual, expected)