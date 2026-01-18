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
@pytest.mark.filterwarnings('ignore:np.find_common_type is deprecated:DeprecationWarning:pyamg.*')
@pytest.mark.parametrize('solver', ['arpack', 'amg', 'lobpcg'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_spectral_eigen_tol_auto(monkeypatch, solver, csr_container):
    """Test that `eigen_tol="auto"` is resolved correctly"""
    if solver == 'amg' and (not pyamg_available):
        pytest.skip('PyAMG is not available.')
    X, _ = make_blobs(n_samples=200, random_state=0, centers=[[1, 1], [-1, -1]], cluster_std=0.01)
    D = pairwise_distances(X)
    S = np.max(D) - D
    solver_func = eigsh if solver == 'arpack' else lobpcg
    default_value = 0 if solver == 'arpack' else None
    if solver == 'amg':
        S = csr_container(S)
    mocked_solver = Mock(side_effect=solver_func)
    monkeypatch.setattr(_spectral_embedding, solver_func.__qualname__, mocked_solver)
    spectral_embedding(S, random_state=42, eigen_solver=solver, eigen_tol='auto')
    mocked_solver.assert_called()
    _, kwargs = mocked_solver.call_args
    assert kwargs['tol'] == default_value