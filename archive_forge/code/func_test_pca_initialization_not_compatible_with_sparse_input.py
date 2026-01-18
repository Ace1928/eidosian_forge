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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pca_initialization_not_compatible_with_sparse_input(csr_container):
    tsne = TSNE(init='pca', learning_rate=100.0, perplexity=1)
    with pytest.raises(TypeError, match='PCA initialization.*'):
        tsne.fit_transform(csr_container([[0, 5], [5, 0]]))