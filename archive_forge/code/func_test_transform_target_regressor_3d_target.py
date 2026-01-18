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
def test_transform_target_regressor_3d_target():
    X = friedman[0]
    y = np.tile(friedman[1].reshape(-1, 1, 1), [1, 3, 2])

    def flatten_data(data):
        return data.reshape(data.shape[0], -1)

    def unflatten_data(data):
        return data.reshape(data.shape[0], -1, 2)
    transformer = FunctionTransformer(func=flatten_data, inverse_func=unflatten_data)
    regr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=transformer)
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape