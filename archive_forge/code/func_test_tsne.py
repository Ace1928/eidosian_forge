import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_tsne():
    n_iter = 250
    perplexity = 5
    n_neighbors = int(3.0 * perplexity + 1)
    rng = np.random.RandomState(0)
    X = rng.randn(20, 2)
    for metric in ['minkowski', 'sqeuclidean']:
        est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance', metric=metric), TSNE(init='random', metric='precomputed', perplexity=perplexity, method='barnes_hut', random_state=42, n_iter=n_iter))
        est_compact = TSNE(init='random', metric=metric, perplexity=perplexity, n_iter=n_iter, method='barnes_hut', random_state=42)
        Xt_chain = est_chain.fit_transform(X)
        Xt_compact = est_compact.fit_transform(X)
        assert_array_almost_equal(Xt_chain, Xt_compact)