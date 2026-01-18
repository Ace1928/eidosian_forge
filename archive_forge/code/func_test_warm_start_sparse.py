import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('Cls', GRADIENT_BOOSTING_ESTIMATORS)
@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_warm_start_sparse(Cls, sparse_container):
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est_dense = Cls(n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True)
    est_dense.fit(X, y)
    est_dense.predict(X)
    est_dense.set_params(n_estimators=200)
    est_dense.fit(X, y)
    y_pred_dense = est_dense.predict(X)
    X_sparse = sparse_container(X)
    est_sparse = Cls(n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True)
    est_sparse.fit(X_sparse, y)
    est_sparse.predict(X)
    est_sparse.set_params(n_estimators=200)
    est_sparse.fit(X_sparse, y)
    y_pred_sparse = est_sparse.predict(X)
    assert_array_almost_equal(est_dense.oob_improvement_[:100], est_sparse.oob_improvement_[:100])
    assert est_dense.oob_scores_[-1] == pytest.approx(est_dense.oob_score_)
    assert_array_almost_equal(est_dense.oob_scores_[:100], est_sparse.oob_scores_[:100])
    assert est_sparse.oob_scores_[-1] == pytest.approx(est_sparse.oob_score_)
    assert_array_almost_equal(y_pred_dense, y_pred_sparse)