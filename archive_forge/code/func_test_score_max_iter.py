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
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_score_max_iter(Estimator, global_random_seed):
    X = np.random.RandomState(global_random_seed).randn(100, 10)
    km1 = Estimator(n_init=1, random_state=global_random_seed, max_iter=1)
    s1 = km1.fit(X).score(X)
    km2 = Estimator(n_init=1, random_state=global_random_seed, max_iter=10)
    s2 = km2.fit(X).score(X)
    assert s2 > s1