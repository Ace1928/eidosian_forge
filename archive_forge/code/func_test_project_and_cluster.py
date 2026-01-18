import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_project_and_cluster(global_random_seed, csr_container):
    model = SpectralBiclustering(random_state=global_random_seed)
    data = np.array([[1, 1, 1], [1, 1, 1], [3, 6, 3], [3, 6, 3]])
    vectors = np.array([[1, 0], [0, 1], [0, 0]])
    for mat in (data, csr_container(data)):
        labels = model._project_and_cluster(mat, vectors, n_clusters=2)
        assert_almost_equal(v_measure_score(labels, [0, 0, 1, 1]), 1.0)