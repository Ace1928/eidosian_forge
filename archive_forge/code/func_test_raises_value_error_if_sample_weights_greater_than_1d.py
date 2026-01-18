import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
def test_raises_value_error_if_sample_weights_greater_than_1d():
    n_sampless = [2, 3]
    n_featuress = [3, 2]
    for n_samples, n_features in zip(n_sampless, n_featuress):
        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        scaler = StandardScaler()
        sample_weight_notOK = rng.randn(n_samples, 1) ** 2
        with pytest.raises(ValueError):
            scaler.fit(X, y, sample_weight=sample_weight_notOK)