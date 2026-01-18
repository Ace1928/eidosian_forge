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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_predict_does_not_change_cluster_centers(csr_container):
    """Check that predict does not change cluster centers.

    Non-regression test for gh-24253.
    """
    X, _ = make_blobs(n_samples=200, n_features=10, centers=10, random_state=0)
    if csr_container is not None:
        X = csr_container(X)
    kmeans = KMeans()
    y_pred1 = kmeans.fit_predict(X)
    kmeans.cluster_centers_ = create_memmap_backed_data(kmeans.cluster_centers_)
    kmeans.labels_ = create_memmap_backed_data(kmeans.labels_)
    y_pred2 = kmeans.predict(X)
    assert_array_equal(y_pred1, y_pred2)