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
def test_sample_weight_unchanged(Estimator):
    X = np.array([[1], [2], [4]])
    sample_weight = np.array([0.5, 0.2, 0.3])
    Estimator(n_clusters=2, random_state=0).fit(X, sample_weight=sample_weight)
    assert_array_equal(sample_weight, np.array([0.5, 0.2, 0.3]))