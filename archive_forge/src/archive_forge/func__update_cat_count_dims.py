import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.special import logsumexp
from .base import BaseEstimator, ClassifierMixin, _fit_context
from .preprocessing import LabelBinarizer, binarize, label_binarize
from .utils._param_validation import Interval
from .utils.extmath import safe_sparse_dot
from .utils.multiclass import _check_partial_fit_first_call
from .utils.validation import _check_sample_weight, check_is_fitted, check_non_negative
def _update_cat_count_dims(cat_count, highest_feature):
    diff = highest_feature + 1 - cat_count.shape[1]
    if diff > 0:
        return np.pad(cat_count, [(0, 0), (0, diff)], 'constant')
    return cat_count