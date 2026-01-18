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
def test_transform_target_regressor_ensure_y_array():
    X, y = friedman
    tt = TransformedTargetRegressor(transformer=DummyCheckerArrayTransformer(), regressor=DummyCheckerListRegressor(), check_inverse=False)
    tt.fit(X.tolist(), y.tolist())
    tt.predict(X.tolist())
    with pytest.raises(AssertionError):
        tt.fit(X, y.tolist())
    with pytest.raises(AssertionError):
        tt.predict(X)