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
@pytest.mark.parametrize('X,y', [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)])
def test_transform_target_regressor_2d_transformer(X, y):
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=transformer)
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    if y.ndim == 1:
        y_tran = regr.transformer_.transform(y.reshape(-1, 1))
    else:
        y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran.squeeze())
    assert y.shape == y_pred.shape
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    lr = LinearRegression()
    transformer2 = clone(transformer)
    if y.ndim == 1:
        lr.fit(X, transformer2.fit_transform(y.reshape(-1, 1)).squeeze())
        y_lr_pred = lr.predict(X).reshape(-1, 1)
        y_pred2 = transformer2.inverse_transform(y_lr_pred).squeeze()
    else:
        lr.fit(X, transformer2.fit_transform(y))
        y_lr_pred = lr.predict(X)
        y_pred2 = transformer2.inverse_transform(y_lr_pred)
    assert_allclose(y_pred, y_pred2)
    assert_allclose(regr.regressor_.coef_, lr.coef_)