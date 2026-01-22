import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
class StepSelector(SelectorMixin, BaseEstimator):
    """Retain every `step` features (beginning with 0).

    If `step < 1`, then no features are selected.
    """

    def __init__(self, step=2):
        self.step = step

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse='csc')
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        if self.step >= 1:
            mask[::self.step] = True
        return mask