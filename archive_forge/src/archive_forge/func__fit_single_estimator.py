from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, _print_elapsed_time, check_random_state
from ..utils._tags import _safe_tags
from ..utils.metaestimators import _BaseComposition
def _fit_single_estimator(estimator, X, y, sample_weight=None, message_clsname=None, message=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError('Underlying estimator {} does not support sample weights.'.format(estimator.__class__.__name__)) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y)
    return estimator