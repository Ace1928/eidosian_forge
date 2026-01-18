import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('Model', [Lasso, ElasticNet, LassoCV, ElasticNetCV])
@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('n_samples, n_features', [(24, 6), (6, 24)])
@pytest.mark.parametrize('with_sample_weight', [True, False])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_sparse_dense_equality(Model, fit_intercept, n_samples, n_features, with_sample_weight, csc_container):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, effective_rank=n_features // 2, n_informative=n_features // 2, bias=4 * fit_intercept, noise=1, random_state=42)
    if with_sample_weight:
        sw = np.abs(np.random.RandomState(42).normal(scale=10, size=y.shape))
    else:
        sw = None
    Xs = csc_container(X)
    params = {'fit_intercept': fit_intercept}
    reg_dense = Model(**params).fit(X, y, sample_weight=sw)
    reg_sparse = Model(**params).fit(Xs, y, sample_weight=sw)
    if fit_intercept:
        assert reg_sparse.intercept_ == pytest.approx(reg_dense.intercept_)
        assert np.average(reg_sparse.predict(X), weights=sw) == pytest.approx(np.average(y, weights=sw))
    assert_allclose(reg_sparse.coef_, reg_dense.coef_)