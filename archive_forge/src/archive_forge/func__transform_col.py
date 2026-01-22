import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
def _transform_col(self, X_col, quantiles, inverse):
    """Private function to transform a single feature."""
    output_distribution = self.output_distribution
    if not inverse:
        lower_bound_x = quantiles[0]
        upper_bound_x = quantiles[-1]
        lower_bound_y = 0
        upper_bound_y = 1
    else:
        lower_bound_x = 0
        upper_bound_x = 1
        lower_bound_y = quantiles[0]
        upper_bound_y = quantiles[-1]
        with np.errstate(invalid='ignore'):
            if output_distribution == 'normal':
                X_col = stats.norm.cdf(X_col)
    with np.errstate(invalid='ignore'):
        if output_distribution == 'normal':
            lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
            upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
        if output_distribution == 'uniform':
            lower_bounds_idx = X_col == lower_bound_x
            upper_bounds_idx = X_col == upper_bound_x
    isfinite_mask = ~np.isnan(X_col)
    X_col_finite = X_col[isfinite_mask]
    if not inverse:
        X_col[isfinite_mask] = 0.5 * (np.interp(X_col_finite, quantiles, self.references_) - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1]))
    else:
        X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)
    X_col[upper_bounds_idx] = upper_bound_y
    X_col[lower_bounds_idx] = lower_bound_y
    if not inverse:
        with np.errstate(invalid='ignore'):
            if output_distribution == 'normal':
                X_col = stats.norm.ppf(X_col)
                clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                X_col = np.clip(X_col, clip_min, clip_max)
    return X_col