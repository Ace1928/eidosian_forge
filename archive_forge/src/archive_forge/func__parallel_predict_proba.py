import itertools
import numbers
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn
import numpy as np
from ..base import ClassifierMixin, RegressorMixin, _fit_context
from ..metrics import accuracy_score, r2_score
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_random_state, column_or_1d, indices_to_mask
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_sample_weight, check_is_fitted, has_fit_parameter
from ._base import BaseEnsemble, _partition_estimators
def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))
    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, 'predict_proba'):
            proba_estimator = estimator.predict_proba(X[:, features])
            if n_classes == len(estimator.classes_):
                proba += proba_estimator
            else:
                proba[:, estimator.classes_] += proba_estimator[:, range(len(estimator.classes_))]
        else:
            predictions = estimator.predict(X[:, features])
            for i in range(n_samples):
                proba[i, predictions[i]] += 1
    return proba