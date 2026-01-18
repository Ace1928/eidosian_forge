import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_select_percentile_classif_sparse(csr_container):
    X, y = make_classification(n_samples=200, n_features=20, n_informative=3, n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, flip_y=0.0, class_sep=10, shuffle=False, random_state=0)
    X = csr_container(X)
    univariate_filter = SelectPercentile(f_classif, percentile=25)
    X_r = univariate_filter.fit(X, y).transform(X)
    X_r2 = GenericUnivariateSelect(f_classif, mode='percentile', param=25).fit(X, y).transform(X)
    assert_array_equal(X_r.toarray(), X_r2.toarray())
    support = univariate_filter.get_support()
    gtruth = np.zeros(20)
    gtruth[:5] = 1
    assert_array_equal(support, gtruth)
    X_r2inv = univariate_filter.inverse_transform(X_r2)
    assert sparse.issparse(X_r2inv)
    support_mask = safe_mask(X_r2inv, support)
    assert X_r2inv.shape == X.shape
    assert_array_equal(X_r2inv[:, support_mask].toarray(), X_r.toarray())
    assert X_r2inv.nnz == X_r.nnz