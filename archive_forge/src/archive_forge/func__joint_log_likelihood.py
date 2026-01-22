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
def _joint_log_likelihood(self, X):
    self._check_n_features(X, reset=False)
    jll = np.zeros((X.shape[0], self.class_count_.shape[0]))
    for i in range(self.n_features_in_):
        indices = X[:, i]
        jll += self.feature_log_prob_[i][:, indices].T
    total_ll = jll + self.class_log_prior_
    return total_ll