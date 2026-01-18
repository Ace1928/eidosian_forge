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
def test_skip_num_points_gradient(csr_container):
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array([[6.080564e-05, -7.120823e-05], [-0.0001718945, -4.000536e-05], [-0.000227172, 8.66331e-05], [-0.0001032577, -3.582033e-05]])
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [1, 0, 3], [1, 2, 0]])
    grad_output = np.array([[0.0, 0.0], [0.0, 0.0], [4.24275173e-08, -3.69569698e-08], [-2.58720939e-09, 7.52706374e-09]])
    _run_answer_test(pos_input, pos_output, neighbors, grad_output, csr_container, False, 0.1, 2)