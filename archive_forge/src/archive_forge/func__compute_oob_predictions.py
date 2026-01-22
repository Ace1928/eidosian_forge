import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
def _compute_oob_predictions(self, X, y):
    """Compute and set the OOB score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or                 (n_samples, 1, n_outputs)
            The OOB predictions.
        """
    if issparse(X):
        X = X.tocsr()
    n_samples = y.shape[0]
    n_outputs = self.n_outputs_
    if is_classifier(self) and hasattr(self, 'n_classes_'):
        oob_pred_shape = (n_samples, self.n_classes_[0], n_outputs)
    else:
        oob_pred_shape = (n_samples, 1, n_outputs)
    oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
    n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)
    n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)
    for estimator in self.estimators_:
        unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
        y_pred = self._get_oob_predictions(estimator, X[unsampled_indices, :])
        oob_pred[unsampled_indices, ...] += y_pred
        n_oob_pred[unsampled_indices, :] += 1
    for k in range(n_outputs):
        if (n_oob_pred == 0).any():
            warn('Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.', UserWarning)
            n_oob_pred[n_oob_pred == 0] = 1
        oob_pred[..., k] /= n_oob_pred[..., [k]]
    return oob_pred