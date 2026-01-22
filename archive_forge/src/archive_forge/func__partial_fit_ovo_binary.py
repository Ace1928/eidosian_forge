import array
import itertools
import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from .base import (
from .metrics.pairwise import pairwise_distances_argmin
from .preprocessing import LabelBinarizer
from .utils import check_random_state
from .utils._param_validation import HasMethods, Interval
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _safe_split, available_if
from .utils.multiclass import (
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, _num_samples, check_is_fitted
def _partial_fit_ovo_binary(estimator, X, y, i, j, partial_fit_params):
    """Partially fit a single binary estimator(one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    if len(y) != 0:
        y_binary = np.zeros_like(y)
        y_binary[y == j] = 1
        partial_fit_params_subset = _check_method_params(X, params=partial_fit_params, indices=cond)
        return _partial_fit_binary(estimator, X[cond], y_binary, partial_fit_params=partial_fit_params_subset)
    return estimator