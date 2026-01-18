from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances
def test_normalize_metric_warning():
    """
    Test that a UserWarning is emitted when using normalized stress with
    metric-MDS.
    """
    msg = 'Normalized stress is not supported'
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    with pytest.raises(ValueError, match=msg):
        mds.smacof(sim, metric=True, normalized_stress=True)