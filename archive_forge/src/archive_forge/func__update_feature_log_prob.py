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
def _update_feature_log_prob(self, alpha):
    feature_log_prob = []
    for i in range(self.n_features_in_):
        smoothed_cat_count = self.category_count_[i] + alpha
        smoothed_class_count = smoothed_cat_count.sum(axis=1)
        feature_log_prob.append(np.log(smoothed_cat_count) - np.log(smoothed_class_count.reshape(-1, 1)))
    self.feature_log_prob_ = feature_log_prob