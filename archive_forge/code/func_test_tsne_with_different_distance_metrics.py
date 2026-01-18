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
@pytest.mark.parametrize('metric, dist_func', [('manhattan', manhattan_distances), ('cosine', cosine_distances)])
@pytest.mark.parametrize('method', ['barnes_hut', 'exact'])
def test_tsne_with_different_distance_metrics(metric, dist_func, method):
    """Make sure that TSNE works for different distance metrics"""
    if method == 'barnes_hut' and metric == 'manhattan':
        pytest.xfail("Distance computations are different for method == 'barnes_hut' and metric == 'manhattan', but this is expected.")
    random_state = check_random_state(0)
    n_components_original = 3
    n_components_embedding = 2
    X = random_state.randn(50, n_components_original).astype(np.float32)
    X_transformed_tsne = TSNE(metric=metric, method=method, n_components=n_components_embedding, random_state=0, n_iter=300, init='random', learning_rate='auto').fit_transform(X)
    X_transformed_tsne_precomputed = TSNE(metric='precomputed', method=method, n_components=n_components_embedding, random_state=0, n_iter=300, init='random', learning_rate='auto').fit_transform(dist_func(X))
    assert_array_equal(X_transformed_tsne, X_transformed_tsne_precomputed)