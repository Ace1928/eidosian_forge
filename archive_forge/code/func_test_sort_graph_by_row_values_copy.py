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
def test_sort_graph_by_row_values_copy(csr_container):
    X_ = csr_container(np.abs(np.random.RandomState(42).randn(10, 10)))
    assert not _is_sorted_by_data(X_)
    X = X_.copy()
    assert sort_graph_by_row_values(X).data is X.data
    X = X_.copy()
    assert sort_graph_by_row_values(X, copy=False).data is X.data
    X = X_.copy()
    assert sort_graph_by_row_values(X, copy=True).data is not X.data
    X = X_.copy()
    assert _check_precomputed(X).data is not X.data
    sort_graph_by_row_values(X.tocsc(), copy=True)
    with pytest.raises(ValueError, match='Use copy=True to allow the conversion'):
        sort_graph_by_row_values(X.tocsc(), copy=False)