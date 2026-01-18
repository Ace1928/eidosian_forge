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
def test_quantile_loss(global_random_seed):
    clf_quantile = GradientBoostingRegressor(n_estimators=100, loss='quantile', max_depth=4, alpha=0.5, random_state=global_random_seed)
    clf_quantile.fit(X_reg, y_reg)
    y_quantile = clf_quantile.predict(X_reg)
    clf_ae = GradientBoostingRegressor(n_estimators=100, loss='absolute_error', max_depth=4, random_state=global_random_seed)
    clf_ae.fit(X_reg, y_reg)
    y_ae = clf_ae.predict(X_reg)
    assert_allclose(y_quantile, y_ae)