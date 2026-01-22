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
def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
    """Implement a single boost using the SAMME discrete algorithm."""
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    y_predict = estimator.predict(X)
    if iboost == 0:
        self.classes_ = getattr(estimator, 'classes_', None)
        self.n_classes_ = len(self.classes_)
    incorrect = y_predict != y
    estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
    if estimator_error <= 0:
        return (sample_weight, 1.0, 0.0)
    n_classes = self.n_classes_
    if estimator_error >= 1.0 - 1.0 / n_classes:
        self.estimators_.pop(-1)
        if len(self.estimators_) == 0:
            raise ValueError('BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit.')
        return (None, None, None)
    estimator_weight = self.learning_rate * (np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0))
    if not iboost == self.n_estimators - 1:
        sample_weight = np.exp(np.log(sample_weight) + estimator_weight * incorrect * (sample_weight > 0))
    return (sample_weight, estimator_weight, estimator_error)