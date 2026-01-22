import warnings
from numbers import Integral
import numpy as np
from sklearn.neighbors._base import _check_precomputed
from ..base import ClassifierMixin, _fit_context
from ..metrics._pairwise_distances_reduction import (
from ..utils._param_validation import StrOptions
from ..utils.arrayfuncs import _all_with_any_reduction_axis_1
from ..utils.extmath import weighted_mode
from ..utils.fixes import _mode
from ..utils.validation import _is_arraylike, _num_samples, check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin, _get_weights
def _adjusted_metric(metric, metric_kwargs, p=None):
    metric_kwargs = metric_kwargs or {}
    if metric == 'minkowski':
        metric_kwargs['p'] = p
        if p == 2:
            metric = 'euclidean'
    return (metric, metric_kwargs)