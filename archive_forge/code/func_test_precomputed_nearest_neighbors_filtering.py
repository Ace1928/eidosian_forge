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
def test_precomputed_nearest_neighbors_filtering():
    n_neighbors = 2
    results = []
    for additional_neighbors in [0, 10]:
        nn = NearestNeighbors(n_neighbors=n_neighbors + additional_neighbors).fit(S)
        graph = nn.kneighbors_graph(S, mode='connectivity')
        embedding = SpectralEmbedding(random_state=0, n_components=2, affinity='precomputed_nearest_neighbors', n_neighbors=n_neighbors).fit(graph).embedding_
        results.append(embedding)
    assert_array_equal(results[0], results[1])