import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
@pytest.mark.parametrize('dim', range(1, 10))
def test_random_superposition(dim):
    state = random_superposition(dim)
    assert dim == len(state)
    assert np.isclose(np.linalg.norm(state), 1.0)