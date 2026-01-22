import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
def _get_support_mask(self):
    mask = np.zeros(self.n_features_in_, dtype=bool)
    if self.step >= 1:
        mask[::self.step] = True
    return mask