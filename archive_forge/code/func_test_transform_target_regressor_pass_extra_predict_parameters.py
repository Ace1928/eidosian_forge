import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils._testing import assert_allclose, assert_no_warnings
def test_transform_target_regressor_pass_extra_predict_parameters():
    X, y = friedman
    regr = TransformedTargetRegressor(regressor=DummyRegressorWithExtraPredictParams(), transformer=DummyTransformer())
    regr.fit(X, y)
    regr.predict(X, check_input=False)
    assert regr.regressor_.predict_called