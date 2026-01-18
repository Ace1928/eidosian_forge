import pickle
import re
import numpy as np
import pytest
from scipy.linalg import LinAlgError
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster._spectral import cluster_qr, discretize
from sklearn.datasets import make_blobs
from sklearn.feature_extraction import img_to_graph
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
@pytest.mark.parametrize('assign_labels', ('kmeans', 'discretize', 'cluster_qr'))
def test_spectral_clustering_sparse(assign_labels, coo_container):
    X, y = make_blobs(n_samples=20, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01)
    S = rbf_kernel(X, gamma=1)
    S = np.maximum(S - 0.0001, 0)
    S = coo_container(S)
    labels = SpectralClustering(random_state=0, n_clusters=2, affinity='precomputed', assign_labels=assign_labels).fit(S).labels_
    assert adjusted_rand_score(y, labels) == 1