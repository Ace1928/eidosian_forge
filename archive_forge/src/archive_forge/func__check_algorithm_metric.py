import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def _check_algorithm_metric(self):
    if self.algorithm == 'auto':
        if self.metric == 'precomputed':
            alg_check = 'brute'
        elif callable(self.metric) or self.metric in VALID_METRICS['ball_tree'] or isinstance(self.metric, DistanceMetric):
            alg_check = 'ball_tree'
        else:
            alg_check = 'brute'
    else:
        alg_check = self.algorithm
    if callable(self.metric):
        if self.algorithm == 'kd_tree':
            raise ValueError("kd_tree does not support callable metric '%s'Function call overhead will resultin very poor performance." % self.metric)
    elif self.metric not in VALID_METRICS[alg_check] and (not isinstance(self.metric, DistanceMetric)):
        raise ValueError("Metric '%s' not valid. Use sorted(sklearn.neighbors.VALID_METRICS['%s']) to get valid options. Metric can also be a callable function." % (self.metric, alg_check))
    if self.metric_params is not None and 'p' in self.metric_params:
        if self.p is not None:
            warnings.warn('Parameter p is found in metric_params. The corresponding parameter from __init__ is ignored.', SyntaxWarning, stacklevel=3)