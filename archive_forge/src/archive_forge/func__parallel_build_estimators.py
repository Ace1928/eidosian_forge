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
def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight, seeds, total_n_estimators, verbose, check_input):
    """Private function used to build a batch of estimators within a job."""
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.estimator_, 'sample_weight')
    has_check_input = has_fit_parameter(ensemble.estimator_, 'check_input')
    requires_feature_indexing = bootstrap_features or max_features != n_features
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")
    estimators = []
    estimators_features = []
    for i in range(n_estimators):
        if verbose > 1:
            print('Building estimator %d of %d for this parallel run (total %d)...' % (i + 1, n_estimators, total_n_estimators))
        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False, random_state=random_state)
        if has_check_input:
            estimator_fit = partial(estimator.fit, check_input=check_input)
        else:
            estimator_fit = estimator.fit
        features, indices = _generate_bagging_indices(random_state, bootstrap_features, bootstrap, n_features, n_samples, max_features, max_samples)
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()
            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0
            X_ = X[:, features] if requires_feature_indexing else X
            estimator_fit(X_, y, sample_weight=curr_sample_weight)
        else:
            X_ = X[indices][:, features] if requires_feature_indexing else X[indices]
            estimator_fit(X_, y[indices])
        estimators.append(estimator)
        estimators_features.append(features)
    return (estimators, estimators_features)