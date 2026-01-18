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
def test_tsne_with_mahalanobis_distance():
    """Make sure that method_parameters works with mahalanobis distance."""
    random_state = check_random_state(0)
    n_samples, n_features = (300, 10)
    X = random_state.randn(n_samples, n_features)
    default_params = {'perplexity': 40, 'n_iter': 250, 'learning_rate': 'auto', 'init': 'random', 'n_components': 3, 'random_state': 0}
    tsne = TSNE(metric='mahalanobis', **default_params)
    msg = 'Must provide either V or VI for Mahalanobis distance'
    with pytest.raises(ValueError, match=msg):
        tsne.fit_transform(X)
    precomputed_X = squareform(pdist(X, metric='mahalanobis'), checks=True)
    X_trans_expected = TSNE(metric='precomputed', **default_params).fit_transform(precomputed_X)
    X_trans = TSNE(metric='mahalanobis', metric_params={'V': np.cov(X.T)}, **default_params).fit_transform(X)
    assert_allclose(X_trans, X_trans_expected)