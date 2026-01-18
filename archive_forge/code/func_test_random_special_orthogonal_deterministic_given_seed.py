import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_random_special_orthogonal_deterministic_given_seed():
    o1 = random_special_orthogonal(2, random_state=1234)
    o2 = random_special_orthogonal(2, random_state=1234)
    np.testing.assert_equal(o1, o2)