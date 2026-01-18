from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances
def test_smacof_error():
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    with pytest.raises(ValueError):
        mds.smacof(sim)
    sim = np.array([[0, 5, 9, 4], [5, 0, 2, 2], [4, 2, 1, 0]])
    with pytest.raises(ValueError):
        mds.smacof(sim)
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    Z = np.array([[-0.266, -0.539], [0.016, -0.238], [-0.2, 0.524]])
    with pytest.raises(ValueError):
        mds.smacof(sim, init=Z, n_init=1)