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
@pytest.mark.parametrize('eigen_solver', ['arpack', 'lobpcg', pytest.param('amg', marks=skip_if_no_pyamg)])
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
def test_spectral_embedding_precomputed_affinity(sparse_container, eigen_solver, dtype, seed=36):
    gamma = 1.0
    X = S if sparse_container is None else sparse_container(S)
    se_precomp = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=np.random.RandomState(seed), eigen_solver=eigen_solver)
    se_rbf = SpectralEmbedding(n_components=2, affinity='rbf', gamma=gamma, random_state=np.random.RandomState(seed), eigen_solver=eigen_solver)
    embed_precomp = se_precomp.fit_transform(rbf_kernel(X.astype(dtype), gamma=gamma))
    embed_rbf = se_rbf.fit_transform(X.astype(dtype))
    assert_array_almost_equal(se_precomp.affinity_matrix_, se_rbf.affinity_matrix_)
    _assert_equal_with_sign_flipping(embed_precomp, embed_rbf, 0.05)