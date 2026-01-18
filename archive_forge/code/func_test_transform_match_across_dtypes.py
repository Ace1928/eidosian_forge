import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_transform_match_across_dtypes(global_random_seed):
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=global_random_seed)
    brc = Birch(n_clusters=4, threshold=1.1)
    Y_64 = brc.fit_transform(X)
    Y_32 = brc.fit_transform(X.astype(np.float32))
    assert_allclose(Y_64, Y_32, atol=1e-06)