import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
def _dense_predict(self, X):
    X = self._compute_kernel(X)
    if X.ndim == 1:
        X = check_array(X, order='C', accept_large_sparse=False)
    kernel = self.kernel
    if callable(self.kernel):
        kernel = 'precomputed'
        if X.shape[1] != self.shape_fit_[0]:
            raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
    svm_type = LIBSVM_IMPL.index(self._impl)
    return libsvm.predict(X, self.support_, self.support_vectors_, self._n_support, self._dual_coef_, self._intercept_, self._probA, self._probB, svm_type=svm_type, kernel=kernel, degree=self.degree, coef0=self.coef0, gamma=self._gamma, cache_size=self.cache_size)