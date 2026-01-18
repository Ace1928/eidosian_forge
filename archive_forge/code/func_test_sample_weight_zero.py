import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('init', ['k-means++', 'random'])
def test_sample_weight_zero(init, global_random_seed):
    """Check that if sample weight is 0, this sample won't be chosen.

    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so
    it's enough to check for KMeans.
    """
    rng = np.random.RandomState(global_random_seed)
    X, _ = make_blobs(n_samples=100, n_features=5, centers=5, random_state=global_random_seed)
    sample_weight = rng.uniform(size=X.shape[0])
    sample_weight[::2] = 0
    x_squared_norms = row_norms(X, squared=True)
    kmeans = KMeans()
    clusters_weighted = kmeans._init_centroids(X=X, x_squared_norms=x_squared_norms, init=init, sample_weight=sample_weight, n_centroids=10, random_state=np.random.RandomState(global_random_seed))
    d = euclidean_distances(X[::2], clusters_weighted)
    assert not np.any(np.isclose(d, 0))