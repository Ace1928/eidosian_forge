from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import get_scorer_names
from ..model_selection import check_cv, cross_val_score
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask):
    candidate_feature_indices = np.flatnonzero(~current_mask)
    scores = {}
    for feature_idx in candidate_feature_indices:
        candidate_mask = current_mask.copy()
        candidate_mask[feature_idx] = True
        if self.direction == 'backward':
            candidate_mask = ~candidate_mask
        X_new = X[:, candidate_mask]
        scores[feature_idx] = cross_val_score(estimator, X_new, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs).mean()
    new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
    return (new_feature_idx, scores[new_feature_idx])