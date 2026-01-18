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
@ignore_warnings(category=EfficiencyWarning)
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + LIL_CONTAINERS)
def test_sparse_precomputed_distance(sparse_container):
    """Make sure that TSNE works identically for sparse and dense matrix"""
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    D_sparse = kneighbors_graph(X, n_neighbors=100, mode='distance', include_self=True)
    D = pairwise_distances(X)
    assert sp.issparse(D_sparse)
    assert_almost_equal(D_sparse.toarray(), D)
    tsne = TSNE(metric='precomputed', random_state=0, init='random', learning_rate='auto')
    Xt_dense = tsne.fit_transform(D)
    Xt_sparse = tsne.fit_transform(sparse_container(D_sparse))
    assert_almost_equal(Xt_dense, Xt_sparse)