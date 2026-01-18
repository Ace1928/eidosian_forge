import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_subcluster_dtype(global_dtype):
    X = make_blobs(n_samples=80, n_features=4, random_state=0)[0].astype(global_dtype, copy=False)
    brc = Birch(n_clusters=4)
    assert brc.fit(X).subcluster_centers_.dtype == global_dtype