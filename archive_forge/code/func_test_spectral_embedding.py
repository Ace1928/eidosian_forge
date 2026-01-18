import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_spectral_embedding():
    n_neighbors = 5
    n_samples = 1000
    centers = np.array([[0.0, 5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 5.0, 1.0]])
    S, true_labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.0, random_state=42)
    est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='connectivity'), SpectralEmbedding(n_neighbors=n_neighbors, affinity='precomputed', random_state=42))
    est_compact = SpectralEmbedding(n_neighbors=n_neighbors, affinity='nearest_neighbors', random_state=42)
    St_compact = est_compact.fit_transform(S)
    St_chain = est_chain.fit_transform(S)
    assert_array_almost_equal(St_chain, St_compact)