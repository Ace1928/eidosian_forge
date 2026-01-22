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
def _fit_regressor(self, X, y, alpha, C, loss, learning_rate, sample_weight, max_iter):
    loss_function = self._get_loss_function(loss)
    penalty_type = self._get_penalty_type(self.penalty)
    learning_rate_type = self._get_learning_rate_type(learning_rate)
    if not hasattr(self, 't_'):
        self.t_ = 1.0
    validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)
    validation_score_cb = self._make_validation_score_cb(validation_mask, X, y, sample_weight)
    random_state = check_random_state(self.random_state)
    seed = random_state.randint(0, MAX_INT)
    dataset, intercept_decay = make_dataset(X, y, sample_weight, random_state=random_state)
    tol = self.tol if self.tol is not None else -np.inf
    if self.average:
        coef = self._standard_coef
        intercept = self._standard_intercept
        average_coef = self._average_coef
        average_intercept = self._average_intercept
    else:
        coef = self.coef_
        intercept = self.intercept_
        average_coef = None
        average_intercept = [0]
    _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
    coef, intercept, average_coef, average_intercept, self.n_iter_ = _plain_sgd(coef, intercept[0], average_coef, average_intercept[0], loss_function, penalty_type, alpha, C, self.l1_ratio, dataset, validation_mask, self.early_stopping, validation_score_cb, int(self.n_iter_no_change), max_iter, tol, int(self.fit_intercept), int(self.verbose), int(self.shuffle), seed, 1.0, 1.0, learning_rate_type, self.eta0, self.power_t, 0, self.t_, intercept_decay, self.average)
    self.t_ += self.n_iter_ * X.shape[0]
    if self.average > 0:
        self._average_intercept = np.atleast_1d(average_intercept)
        self._standard_intercept = np.atleast_1d(intercept)
        if self.average <= self.t_ - 1.0:
            self.coef_ = average_coef
            self.intercept_ = np.atleast_1d(average_intercept)
        else:
            self.coef_ = coef
            self.intercept_ = np.atleast_1d(intercept)
    else:
        self.intercept_ = np.atleast_1d(intercept)