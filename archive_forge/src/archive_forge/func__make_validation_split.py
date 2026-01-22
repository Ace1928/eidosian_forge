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
def _make_validation_split(self, y, sample_mask):
    """Split the dataset between training set and validation set.

        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.

        sample_mask : ndarray of shape (n_samples, )
            A boolean array indicating whether each sample should be included
            for validation set.

        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to True on the validation set, False on the training set.
        """
    n_samples = y.shape[0]
    validation_mask = np.zeros(n_samples, dtype=np.bool_)
    if not self.early_stopping:
        return validation_mask
    if is_classifier(self):
        splitter_type = StratifiedShuffleSplit
    else:
        splitter_type = ShuffleSplit
    cv = splitter_type(test_size=self.validation_fraction, random_state=self.random_state)
    idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
    if not np.any(sample_mask[idx_val]):
        raise ValueError('The sample weights for validation set are all zero, consider using a different random state.')
    if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
        raise ValueError('Splitting %d samples into a train set and a validation set with validation_fraction=%r led to an empty set (%d and %d samples). Please either change validation_fraction, increase number of samples, or disable early_stopping.' % (n_samples, self.validation_fraction, idx_train.shape[0], idx_val.shape[0]))
    validation_mask[idx_val] = True
    return validation_mask