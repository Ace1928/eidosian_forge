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
@pytest.mark.parametrize('array_constr', data_containers, ids=data_containers_ids)
def test_relocate_empty_clusters(array_constr):
    X = np.array([-10.0, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10]).reshape(-1, 1)
    X = array_constr(X)
    sample_weight = np.ones(10)
    centers_old = np.array([-10.0, -10, -10]).reshape(-1, 1)
    centers_new = np.array([-16.5, -10, -10]).reshape(-1, 1)
    weight_in_clusters = np.array([10.0, 0, 0])
    labels = np.zeros(10, dtype=np.int32)
    if array_constr is np.array:
        _relocate_empty_clusters_dense(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels)
    else:
        _relocate_empty_clusters_sparse(X.data, X.indices, X.indptr, sample_weight, centers_old, centers_new, weight_in_clusters, labels)
    assert_array_equal(weight_in_clusters, [8, 1, 1])
    assert_allclose(centers_new, [[-36], [10], [9.5]])