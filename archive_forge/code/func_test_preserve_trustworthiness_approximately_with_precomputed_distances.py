import sys
from io import StringIO
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from scipy.spatial.distance import pdist, squareform
from sklearn import config_context
from sklearn.datasets import make_blobs
from sklearn.exceptions import EfficiencyWarning
from sklearn.manifold import (  # type: ignore
from sklearn.manifold._t_sne import (
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.metrics.pairwise import (
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
def test_preserve_trustworthiness_approximately_with_precomputed_distances():
    random_state = check_random_state(0)
    for i in range(3):
        X = random_state.randn(80, 2)
        D = squareform(pdist(X), 'sqeuclidean')
        tsne = TSNE(n_components=2, perplexity=2, learning_rate=100.0, early_exaggeration=2.0, metric='precomputed', random_state=i, verbose=0, n_iter=500, init='random')
        X_embedded = tsne.fit_transform(D)
        t = trustworthiness(D, X_embedded, n_neighbors=1, metric='precomputed')
        assert t > 0.95