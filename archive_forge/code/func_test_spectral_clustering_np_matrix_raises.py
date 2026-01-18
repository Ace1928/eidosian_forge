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
def test_spectral_clustering_np_matrix_raises():
    """Check that spectral_clustering raises an informative error when passed
    a np.matrix. See #10993"""
    X = np.matrix([[0.0, 2.0], [2.0, 0.0]])
    msg = 'np\\.matrix is not supported. Please convert to a numpy array'
    with pytest.raises(TypeError, match=msg):
        spectral_clustering(X)