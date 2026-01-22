import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import xlogy
from ..base import (
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import _safe_indexing, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.extmath import softmax, stable_cumsum
from ..utils.metadata_routing import (
from ..utils.validation import (
from ._base import BaseEnsemble
def _boost_real(self, iboost, X, y, sample_weight, random_state):
    """Implement a single boost using the SAMME.R real algorithm."""
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict_proba = estimator.predict_proba(X)
    if iboost == 0:
        self.classes_ = getattr(estimator, 'classes_', None)
        self.n_classes_ = len(self.classes_)
    y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
    incorrect = y_predict != y
    estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
    if estimator_error <= 0:
        return (sample_weight, 1.0, 0.0)
    n_classes = self.n_classes_
    classes = self.classes_
    y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
    y_coding = y_codes.take(classes == y[:, np.newaxis])
    proba = y_predict_proba
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    estimator_weight = -1.0 * self.learning_rate * ((n_classes - 1.0) / n_classes) * xlogy(y_coding, y_predict_proba).sum(axis=1)
    if not iboost == self.n_estimators - 1:
        sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))
    return (sample_weight, 1.0, estimator_error)