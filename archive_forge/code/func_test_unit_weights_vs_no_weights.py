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
@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_unit_weights_vs_no_weights(Estimator, input_data, global_random_seed):
    sample_weight = np.ones(n_samples)
    km = Estimator(n_clusters=n_clusters, random_state=global_random_seed, n_init=1)
    km_none = clone(km).fit(input_data, sample_weight=None)
    km_ones = clone(km).fit(input_data, sample_weight=sample_weight)
    assert_array_equal(km_none.labels_, km_ones.labels_)
    assert_allclose(km_none.cluster_centers_, km_ones.cluster_centers_)