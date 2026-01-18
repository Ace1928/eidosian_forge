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
def test_transform_target_regressor_route_pipeline():
    X, y = friedman
    regr = TransformedTargetRegressor(regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer())
    estimators = [('normalize', StandardScaler()), ('est', regr)]
    pip = Pipeline(estimators)
    pip.fit(X, y, **{'est__check_input': False})
    assert regr.transformer_.fit_counter == 1