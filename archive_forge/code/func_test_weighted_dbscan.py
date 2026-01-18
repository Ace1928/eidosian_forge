import pickle
import warnings
import numpy as np
import pytest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
def test_weighted_dbscan(global_random_seed):
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2])
    with pytest.raises(ValueError):
        dbscan([[0], [1]], sample_weight=[2, 3, 4])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=None, min_samples=6)[0])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=[5, 5], min_samples=6)[0])
    assert_array_equal([0], dbscan([[0], [1]], sample_weight=[6, 5], min_samples=6)[0])
    assert_array_equal([0, 1], dbscan([[0], [1]], sample_weight=[6, 6], min_samples=6)[0])
    assert_array_equal([0, 1], dbscan([[0], [1]], eps=1.5, sample_weight=[5, 1], min_samples=6)[0])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=[5, 0], eps=1.5, min_samples=6)[0])
    assert_array_equal([0, 1], dbscan([[0], [1]], sample_weight=[5.9, 0.1], eps=1.5, min_samples=6)[0])
    assert_array_equal([0, 1], dbscan([[0], [1]], sample_weight=[6, 0], eps=1.5, min_samples=6)[0])
    assert_array_equal([], dbscan([[0], [1]], sample_weight=[6, -1], eps=1.5, min_samples=6)[0])
    rng = np.random.RandomState(global_random_seed)
    sample_weight = rng.randint(0, 5, X.shape[0])
    core1, label1 = dbscan(X, sample_weight=sample_weight)
    assert len(label1) == len(X)
    X_repeated = np.repeat(X, sample_weight, axis=0)
    core_repeated, label_repeated = dbscan(X_repeated)
    core_repeated_mask = np.zeros(X_repeated.shape[0], dtype=bool)
    core_repeated_mask[core_repeated] = True
    core_mask = np.zeros(X.shape[0], dtype=bool)
    core_mask[core1] = True
    assert_array_equal(np.repeat(core_mask, sample_weight), core_repeated_mask)
    D = pairwise_distances(X)
    core3, label3 = dbscan(D, sample_weight=sample_weight, metric='precomputed')
    assert_array_equal(core1, core3)
    assert_array_equal(label1, label3)
    est = DBSCAN().fit(X, sample_weight=sample_weight)
    core4 = est.core_sample_indices_
    label4 = est.labels_
    assert_array_equal(core1, core4)
    assert_array_equal(label1, label4)
    est = DBSCAN()
    label5 = est.fit_predict(X, sample_weight=sample_weight)
    core5 = est.core_sample_indices_
    assert_array_equal(core1, core5)
    assert_array_equal(label1, label5)
    assert_array_equal(label1, est.labels_)