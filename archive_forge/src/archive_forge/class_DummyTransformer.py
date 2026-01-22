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
class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy transformer which count how many time fit was called."""

    def __init__(self, fit_counter=0):
        self.fit_counter = fit_counter

    def fit(self, X, y=None):
        self.fit_counter += 1
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X