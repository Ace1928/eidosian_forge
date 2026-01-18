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
@pytest.mark.parametrize('GradientBoostingEstimator', GRADIENT_BOOSTING_ESTIMATORS)
def test_oob_scores(GradientBoostingEstimator):
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    estimator = GradientBoostingEstimator(n_estimators=100, random_state=1, subsample=0.5)
    estimator.fit(X, y)
    assert estimator.oob_scores_.shape[0] == 100
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)
    estimator = GradientBoostingEstimator(n_estimators=100, random_state=1, subsample=0.5, n_iter_no_change=5)
    estimator.fit(X, y)
    assert estimator.oob_scores_.shape[0] < 100
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)