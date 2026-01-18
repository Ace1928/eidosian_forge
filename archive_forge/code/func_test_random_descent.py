import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_random_descent(csr_container):
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X, y)
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X, y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X.T, y[:20])
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X.T, y[:20])
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(csr_container(X), y)
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(csr_container(X), y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
    clf_cyclic = MultiTaskElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X, new_y)
    clf_random = MultiTaskElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X, new_y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)