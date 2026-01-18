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
def test_weighted_vs_repeated(global_random_seed):
    sample_weight = np.random.RandomState(global_random_seed).randint(1, 5, size=n_samples)
    X_repeat = np.repeat(X, sample_weight, axis=0)
    km = KMeans(init=centers, n_init=1, n_clusters=n_clusters, random_state=global_random_seed)
    km_weighted = clone(km).fit(X, sample_weight=sample_weight)
    repeated_labels = np.repeat(km_weighted.labels_, sample_weight)
    km_repeated = clone(km).fit(X_repeat)
    assert_array_equal(km_repeated.labels_, repeated_labels)
    assert_allclose(km_weighted.inertia_, km_repeated.inertia_)
    assert_allclose(_sort_centers(km_weighted.cluster_centers_), _sort_centers(km_repeated.cluster_centers_))