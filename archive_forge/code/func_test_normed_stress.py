from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances
@pytest.mark.parametrize('k', [0.5, 1.5, 2])
def test_normed_stress(k):
    """Test that non-metric MDS normalized stress is scale-invariant."""
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    X1, stress1 = mds.smacof(sim, metric=False, max_iter=5, random_state=0)
    X2, stress2 = mds.smacof(k * sim, metric=False, max_iter=5, random_state=0)
    assert_allclose(stress1, stress2, rtol=1e-05)
    assert_allclose(X1, X2, rtol=1e-05)