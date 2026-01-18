import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
@pytest.mark.parametrize('dim', range(1, 10))
def test_random_density_matrix(dim):
    state = random_density_matrix(dim)
    assert state.shape == (dim, dim)
    np.testing.assert_allclose(np.trace(state), 1)
    np.testing.assert_allclose(state, state.T.conj())
    eigs, _ = np.linalg.eigh(state)
    assert np.all(eigs >= 0)