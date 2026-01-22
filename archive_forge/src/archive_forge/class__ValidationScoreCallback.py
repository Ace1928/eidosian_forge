import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
class _ValidationScoreCallback:
    """Callback for early stopping based on validation score"""

    def __init__(self, estimator, X_val, y_val, sample_weight_val, classes=None):
        self.estimator = clone(estimator)
        self.estimator.t_ = 1
        if classes is not None:
            self.estimator.classes_ = classes
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val

    def __call__(self, coef, intercept):
        est = self.estimator
        est.coef_ = coef.reshape(1, -1)
        est.intercept_ = np.atleast_1d(intercept)
        return est.score(self.X_val, self.y_val, self.sample_weight_val)