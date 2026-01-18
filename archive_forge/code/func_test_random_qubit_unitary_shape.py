import numpy as np
import pytest
import cirq
from cirq import value
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def test_random_qubit_unitary_shape():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary((3, 4, 5), True, rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((3 * 4 * 5,), True, rng).ravel()
    np.testing.assert_almost_equal(actual, expected)