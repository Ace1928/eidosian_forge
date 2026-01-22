import math
import warnings
from numbers import Real
import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from .utils import check_array, check_consistent_length
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.validation import _check_sample_weight, check_is_fitted
def _build_y(self, X, y, sample_weight, trim_duplicates=True):
    """Build the y_ IsotonicRegression."""
    self._check_input_data_shape(X)
    X = X.reshape(-1)
    if self.increasing == 'auto':
        self.increasing_ = check_increasing(X, y)
    else:
        self.increasing_ = self.increasing
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    mask = sample_weight > 0
    X, y, sample_weight = (X[mask], y[mask], sample_weight[mask])
    order = np.lexsort((y, X))
    X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
    unique_X, unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)
    X = unique_X
    y = isotonic_regression(unique_y, sample_weight=unique_sample_weight, y_min=self.y_min, y_max=self.y_max, increasing=self.increasing_)
    self.X_min_, self.X_max_ = (np.min(X), np.max(X))
    if trim_duplicates:
        keep_data = np.ones((len(y),), dtype=bool)
        keep_data[1:-1] = np.logical_or(np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:]))
        return (X[keep_data], y[keep_data])
    else:
        return (X, y)