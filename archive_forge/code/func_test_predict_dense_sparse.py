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
@pytest.mark.parametrize('X_csr', X_as_any_csr)
@pytest.mark.parametrize('init', ['random', 'k-means++', centers], ids=['random', 'k-means++', 'ndarray'])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_predict_dense_sparse(Estimator, init, X_csr):
    n_init = 10 if isinstance(init, str) else 1
    km = Estimator(n_clusters=n_clusters, init=init, n_init=n_init, random_state=0)
    km.fit(X_csr)
    assert_array_equal(km.predict(X), km.labels_)
    km.fit(X)
    assert_array_equal(km.predict(X_csr), km.labels_)