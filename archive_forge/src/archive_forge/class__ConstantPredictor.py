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
class _ConstantPredictor(BaseEstimator):
    """Helper predictor to be used when only one class is present."""

    def fit(self, X, y):
        check_params = dict(force_all_finite=False, dtype=None, ensure_2d=False, accept_sparse=True)
        self._validate_data(X, y, reset=True, validate_separately=(check_params, check_params))
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        self._validate_data(X, force_all_finite=False, dtype=None, accept_sparse=True, ensure_2d=False, reset=False)
        return np.repeat(self.y_, _num_samples(X))

    def decision_function(self, X):
        check_is_fitted(self)
        self._validate_data(X, force_all_finite=False, dtype=None, accept_sparse=True, ensure_2d=False, reset=False)
        return np.repeat(self.y_, _num_samples(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        self._validate_data(X, force_all_finite=False, dtype=None, accept_sparse=True, ensure_2d=False, reset=False)
        y_ = self.y_.astype(np.float64)
        return np.repeat([np.hstack([1 - y_, y_])], _num_samples(X), axis=0)