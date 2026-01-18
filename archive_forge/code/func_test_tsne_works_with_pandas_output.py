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
def test_tsne_works_with_pandas_output():
    """Make sure that TSNE works when the output is set to "pandas".

    Non-regression test for gh-25365.
    """
    pytest.importorskip('pandas')
    with config_context(transform_output='pandas'):
        arr = np.arange(35 * 4).reshape(35, 4)
        TSNE(n_components=2).fit_transform(arr)