from unittest.mock import Mock
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
from sklearn.utils.fixes import laplacian as csgraph_laplacian
@pytest.mark.parametrize('sparse_container', [None, *CSR_CONTAINERS])
def test_spectral_embedding_callable_affinity(sparse_container, seed=36):
    gamma = 0.9
    kern = rbf_kernel(S, gamma=gamma)
    X = S if sparse_container is None else sparse_container(S)
    se_callable = SpectralEmbedding(n_components=2, affinity=lambda x: rbf_kernel(x, gamma=gamma), gamma=gamma, random_state=np.random.RandomState(seed))
    se_rbf = SpectralEmbedding(n_components=2, affinity='rbf', gamma=gamma, random_state=np.random.RandomState(seed))
    embed_rbf = se_rbf.fit_transform(X)
    embed_callable = se_callable.fit_transform(X)
    assert_array_almost_equal(se_callable.affinity_matrix_, se_rbf.affinity_matrix_)
    assert_array_almost_equal(kern, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_rbf, embed_callable, 0.05)