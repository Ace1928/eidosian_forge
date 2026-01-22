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
def _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter):
    """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
    validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)
    random_state = check_random_state(self.random_state)
    seeds = random_state.randint(MAX_INT, size=len(self.classes_))
    result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, require='sharedmem')((delayed(fit_binary)(self, i, X, y, alpha, C, learning_rate, max_iter, self._expanded_class_weight[i], 1.0, sample_weight, validation_mask=validation_mask, random_state=seed) for i, seed in enumerate(seeds)))
    n_iter_ = 0.0
    for i, (_, intercept, n_iter_i) in enumerate(result):
        self.intercept_[i] = intercept
        n_iter_ = max(n_iter_, n_iter_i)
    self.t_ += n_iter_ * X.shape[0]
    self.n_iter_ = n_iter_
    if self.average > 0:
        if self.average <= self.t_ - 1.0:
            self.coef_ = self._average_coef
            self.intercept_ = self._average_intercept
        else:
            self.coef_ = self._standard_coef
            self._standard_intercept = np.atleast_1d(self.intercept_)
            self.intercept_ = self._standard_intercept