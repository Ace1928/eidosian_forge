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
def _check_input_data_shape(self, X):
    if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
        msg = 'Isotonic regression input X should be a 1d array or 2d array with 1 feature'
        raise ValueError(msg)