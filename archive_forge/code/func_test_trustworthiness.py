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
def test_trustworthiness():
    random_state = check_random_state(0)
    X = random_state.randn(100, 2)
    assert trustworthiness(X, 5.0 + X / 10.0) == 1.0
    X = np.arange(100).reshape(-1, 1)
    X_embedded = X.copy()
    random_state.shuffle(X_embedded)
    assert trustworthiness(X, X_embedded) < 0.6
    X = np.arange(5).reshape(-1, 1)
    X_embedded = np.array([[0], [2], [4], [1], [3]])
    assert_almost_equal(trustworthiness(X, X_embedded, n_neighbors=1), 0.2)