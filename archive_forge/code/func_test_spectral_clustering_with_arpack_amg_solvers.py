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
@pytest.mark.filterwarnings('ignore:scipy.rand is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.filterwarnings('ignore:`np.float` is a deprecated alias:DeprecationWarning:pyamg.*')
@pytest.mark.filterwarnings('ignore:scipy.linalg.pinv2 is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.filterwarnings('ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*')
def test_spectral_clustering_with_arpack_amg_solvers():
    x, y = np.indices((40, 40))
    center1, center2 = ((14, 12), (20, 25))
    radius1, radius2 = (8, 7)
    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circles = circle1 | circle2
    mask = circles.copy()
    img = circles.astype(float)
    graph = img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())
    labels_arpack = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack', random_state=0)
    assert len(np.unique(labels_arpack)) == 2
    if amg_loaded:
        labels_amg = spectral_clustering(graph, n_clusters=2, eigen_solver='amg', random_state=0)
        assert adjusted_rand_score(labels_arpack, labels_amg) == 1
    else:
        with pytest.raises(ValueError):
            spectral_clustering(graph, n_clusters=2, eigen_solver='amg', random_state=0)