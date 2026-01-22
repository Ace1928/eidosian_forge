import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
class _RidgeClassifierMixin(LinearClassifierMixin):

    def _prepare_data(self, X, y, sample_weight, solver):
        """Validate `X` and `y` and binarize `y`.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        solver : str
            The solver used in `Ridge` to know which sparse format to support.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Validated training data.

        y : ndarray of shape (n_samples,)
            Validated target values.

        sample_weight : ndarray of shape (n_samples,)
            Validated sample weights.

        Y : ndarray of shape (n_samples, n_classes)
            The binarized version of `y`.
        """
        accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), solver)
        X, y = self._validate_data(X, y, accept_sparse=accept_sparse, multi_output=True, y_numeric=False)
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            y = column_or_1d(y, warn=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        if self.class_weight:
            sample_weight = sample_weight * compute_sample_weight(self.class_weight, y)
        return (X, y, sample_weight, Y)

    def predict(self, X):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : {array-like, spare matrix} of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`. In
            a multilabel problem, it returns a matrix of shape
            `(n_samples, n_outputs)`.
        """
        check_is_fitted(self, attributes=['_label_binarizer'])
        if self._label_binarizer.y_type_.startswith('multilabel'):
            scores = 2 * (self.decision_function(X) > 0) - 1
            return self._label_binarizer.inverse_transform(scores)
        return super().predict(X)

    @property
    def classes_(self):
        """Classes labels."""
        return self._label_binarizer.classes_

    def _more_tags(self):
        return {'multilabel': True}