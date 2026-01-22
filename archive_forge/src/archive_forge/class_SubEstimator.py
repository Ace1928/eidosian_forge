import functools
from inspect import signature
import numpy as np
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import (
from sklearn.utils.validation import check_is_fitted
class SubEstimator(BaseEstimator):

    def __init__(self, param=1, hidden_method=None):
        self.param = param
        self.hidden_method = hidden_method

    def fit(self, X, y=None, *args, **kwargs):
        self.coef_ = np.arange(X.shape[1])
        self.classes_ = []
        return True

    def _check_fit(self):
        check_is_fitted(self)

    @hides
    def inverse_transform(self, X, *args, **kwargs):
        self._check_fit()
        return X

    @hides
    def transform(self, X, *args, **kwargs):
        self._check_fit()
        return X

    @hides
    def predict(self, X, *args, **kwargs):
        self._check_fit()
        return np.ones(X.shape[0])

    @hides
    def predict_proba(self, X, *args, **kwargs):
        self._check_fit()
        return np.ones(X.shape[0])

    @hides
    def predict_log_proba(self, X, *args, **kwargs):
        self._check_fit()
        return np.ones(X.shape[0])

    @hides
    def decision_function(self, X, *args, **kwargs):
        self._check_fit()
        return np.ones(X.shape[0])

    @hides
    def score(self, X, y, *args, **kwargs):
        self._check_fit()
        return 1.0