import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.filterwarnings('ignore:EfficiencyWarning')
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_precomputed_sparse_invalid(csr_container):
    dist = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    dist_csr = csr_container(dist)
    neigh = neighbors.NearestNeighbors(n_neighbors=1, metric='precomputed')
    neigh.fit(dist_csr)
    neigh.kneighbors(None, n_neighbors=1)
    neigh.kneighbors(np.array([[0.0, 0.0, 0.0]]), n_neighbors=2)
    dist = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    dist_csr = csr_container(dist)
    neigh.fit(dist_csr)
    msg = '2 neighbors per samples are required, but some samples have only 1'
    with pytest.raises(ValueError, match=msg):
        neigh.kneighbors(None, n_neighbors=1)
    dist = np.array([[5.0, 2.0, 1.0], [-2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    dist_csr = csr_container(dist)
    msg = 'Negative values in data passed to precomputed distance matrix.'
    with pytest.raises(ValueError, match=msg):
        neigh.kneighbors(dist_csr, n_neighbors=1)