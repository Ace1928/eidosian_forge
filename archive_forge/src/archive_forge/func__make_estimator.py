from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
def _make_estimator(self, append=True, random_state=None):
    """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
    estimator = clone(self.estimator_)
    estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
    if random_state is not None:
        _set_random_states(estimator, random_state)
    if append:
        self.estimators_.append(estimator)
    return estimator