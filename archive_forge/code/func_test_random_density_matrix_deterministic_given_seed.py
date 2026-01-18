import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_random_density_matrix_deterministic_given_seed():
    state1 = random_density_matrix(10, random_state=1234)
    state2 = random_density_matrix(10, random_state=1234)
    np.testing.assert_equal(state1, state2)