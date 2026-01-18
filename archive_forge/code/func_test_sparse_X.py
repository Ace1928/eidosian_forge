import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_X(global_random_seed, global_dtype, csr_container):
    X, y = make_blobs(n_samples=100, centers=10, random_state=global_random_seed)
    X = X.astype(global_dtype, copy=False)
    brc = Birch(n_clusters=10)
    brc.fit(X)
    csr = csr_container(X)
    brc_sparse = Birch(n_clusters=10)
    brc_sparse.fit(csr)
    assert brc_sparse.subcluster_centers_.dtype == global_dtype
    assert_array_equal(brc.labels_, brc_sparse.labels_)
    assert_allclose(brc.subcluster_centers_, brc_sparse.subcluster_centers_)