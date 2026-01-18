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
def test_transform_target_regressor_2d_transformer_multioutput():
    X = friedman[0]
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=transformer)
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran)
    assert y.shape == y_pred.shape
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))
    assert_allclose(regr.regressor_.coef_, lr.coef_)