from copy import deepcopy
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from ..exceptions import NotFittedError
from ..utils._param_validation import HasMethods, Interval, Options
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.validation import _num_features, check_is_fitted, check_scalar
from ._base import SelectorMixin, _get_feature_importances
def _calculate_threshold(estimator, importances, threshold):
    """Interpret the threshold value"""
    if threshold is None:
        est_name = estimator.__class__.__name__
        is_l1_penalized = hasattr(estimator, 'penalty') and estimator.penalty == 'l1'
        is_lasso = 'Lasso' in est_name
        is_elasticnet_l1_penalized = 'ElasticNet' in est_name and (hasattr(estimator, 'l1_ratio_') and np.isclose(estimator.l1_ratio_, 1.0) or (hasattr(estimator, 'l1_ratio') and np.isclose(estimator.l1_ratio, 1.0)))
        if is_l1_penalized or is_lasso or is_elasticnet_l1_penalized:
            threshold = 1e-05
        else:
            threshold = 'mean'
    if isinstance(threshold, str):
        if '*' in threshold:
            scale, reference = threshold.split('*')
            scale = float(scale.strip())
            reference = reference.strip()
            if reference == 'median':
                reference = np.median(importances)
            elif reference == 'mean':
                reference = np.mean(importances)
            else:
                raise ValueError('Unknown reference: ' + reference)
            threshold = scale * reference
        elif threshold == 'median':
            threshold = np.median(importances)
        elif threshold == 'mean':
            threshold = np.mean(importances)
        else:
            raise ValueError("Expected threshold='mean' or threshold='median' got %s" % threshold)
    else:
        threshold = float(threshold)
    return threshold