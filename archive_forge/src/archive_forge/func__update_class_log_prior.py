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
def _update_class_log_prior(self, class_prior=None):
    """Update class log priors.

        The class log priors are based on `class_prior`, class count or the
        number of classes. This method is called each time `fit` or
        `partial_fit` update the model.
        """
    n_classes = len(self.classes_)
    if class_prior is not None:
        if len(class_prior) != n_classes:
            raise ValueError('Number of priors must match number of classes.')
        self.class_log_prior_ = np.log(class_prior)
    elif self.fit_prior:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            log_class_count = np.log(self.class_count_)
        self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
    else:
        self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))