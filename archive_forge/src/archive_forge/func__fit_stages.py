import math
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from .._loss.loss import (
from ..base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from ..dummy import DummyClassifier, DummyRegressor
from ..exceptions import NotFittedError
from ..model_selection import train_test_split
from ..preprocessing import LabelEncoder
from ..tree import DecisionTreeRegressor
from ..tree._tree import DOUBLE, DTYPE, TREE_LEAF
from ..utils import check_array, check_random_state, column_or_1d
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.stats import _weighted_percentile
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import BaseEnsemble
from ._gradient_boosting import _random_sample_mask, predict_stage, predict_stages
def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state, X_val, y_val, sample_weight_val, begin_at_stage=0, monitor=None):
    """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
    n_samples = X.shape[0]
    do_oob = self.subsample < 1.0
    sample_mask = np.ones((n_samples,), dtype=bool)
    n_inbag = max(1, int(self.subsample * n_samples))
    if self.verbose:
        verbose_reporter = VerboseReporter(verbose=self.verbose)
        verbose_reporter.init(self, begin_at_stage)
    X_csc = csc_matrix(X) if issparse(X) else None
    X_csr = csr_matrix(X) if issparse(X) else None
    if self.n_iter_no_change is not None:
        loss_history = np.full(self.n_iter_no_change, np.inf)
        y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)
    if isinstance(self._loss, (HalfSquaredError, HalfBinomialLoss)):
        factor = 2
    else:
        factor = 1
    i = begin_at_stage
    for i in range(begin_at_stage, self.n_estimators):
        if do_oob:
            sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
            y_oob_masked = y[~sample_mask]
            sample_weight_oob_masked = sample_weight[~sample_mask]
            if i == 0:
                initial_loss = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
        raw_predictions = self._fit_stage(i, X, y, raw_predictions, sample_weight, sample_mask, random_state, X_csc=X_csc, X_csr=X_csr)
        if do_oob:
            self.train_score_[i] = factor * self._loss(y_true=y[sample_mask], raw_prediction=raw_predictions[sample_mask], sample_weight=sample_weight[sample_mask])
            self.oob_scores_[i] = factor * self._loss(y_true=y_oob_masked, raw_prediction=raw_predictions[~sample_mask], sample_weight=sample_weight_oob_masked)
            previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
            self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
            self.oob_score_ = self.oob_scores_[-1]
        else:
            self.train_score_[i] = factor * self._loss(y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight)
        if self.verbose > 0:
            verbose_reporter.update(i, self)
        if monitor is not None:
            early_stopping = monitor(i, self, locals())
            if early_stopping:
                break
        if self.n_iter_no_change is not None:
            validation_loss = factor * self._loss(y_val, next(y_val_pred_iter), sample_weight_val)
            if np.any(validation_loss + self.tol < loss_history):
                loss_history[i % len(loss_history)] = validation_loss
            else:
                break
    return i + 1