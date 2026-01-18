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
@pytest.mark.parametrize('sample_weight', [True, None])
def test_standard_scaler_trasform_with_partial_fit(sample_weight):
    X = X_2d[:100, :]
    if sample_weight:
        sample_weight = rng.rand(X.shape[0])
    scaler_incr = StandardScaler()
    for i, batch in enumerate(gen_batches(X.shape[0], 1)):
        X_sofar = X[:i + 1, :]
        chunks_copy = X_sofar.copy()
        if sample_weight is None:
            scaled_batch = StandardScaler().fit_transform(X_sofar)
            scaler_incr = scaler_incr.partial_fit(X[batch])
        else:
            scaled_batch = StandardScaler().fit_transform(X_sofar, sample_weight=sample_weight[:i + 1])
            scaler_incr = scaler_incr.partial_fit(X[batch], sample_weight=sample_weight[batch])
        scaled_incr = scaler_incr.transform(X_sofar)
        assert_array_almost_equal(scaled_batch, scaled_incr)
        assert_array_almost_equal(X_sofar, chunks_copy)
        right_input = scaler_incr.inverse_transform(scaled_incr)
        assert_array_almost_equal(X_sofar, right_input)
        zero = np.zeros(X.shape[1])
        epsilon = np.finfo(float).eps
        assert_array_less(zero, scaler_incr.var_ + epsilon)
        assert_array_less(zero, scaler_incr.scale_ + epsilon)
        if sample_weight is None:
            assert i + 1 == scaler_incr.n_samples_seen_
        else:
            assert np.sum(sample_weight[:i + 1]) == pytest.approx(scaler_incr.n_samples_seen_)