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
@pytest.mark.parametrize('algorithm', ['lloyd', 'elkan'])
def test_relocating_with_duplicates(algorithm, array_constr):
    """Check that kmeans stops when there are more centers than non-duplicate samples

    Non-regression test for issue:
    https://github.com/scikit-learn/scikit-learn/issues/28055
    """
    X = np.array([[0, 0], [1, 1], [1, 1], [1, 0], [0, 1]])
    km = KMeans(n_clusters=5, init=X, algorithm=algorithm)
    msg = 'Number of distinct clusters \\(4\\) found smaller than n_clusters \\(5\\)'
    with pytest.warns(ConvergenceWarning, match=msg):
        km.fit(array_constr(X))
    assert km.n_iter_ == 1