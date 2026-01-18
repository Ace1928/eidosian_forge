import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
def test_split_interaction_constraints():
    """Check that allowed_features are respected."""
    n_features = 4
    allowed_features = np.array([0, 3], dtype=np.uint32)
    n_bins = 5
    n_samples = 10
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_hessians = n_samples
    hessians_are_constant = True
    split_features = []
    for i in range(10):
        rng = np.random.RandomState(919 + i)
        X_binned = np.asfortranarray(rng.randint(0, n_bins - 1, size=(n_samples, n_features)), dtype=X_BINNED_DTYPE)
        X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)
        all_gradients = (10 * X_binned[:, 1] + rng.randn(n_samples)).astype(G_H_DTYPE)
        sum_gradients = all_gradients.sum()
        builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
        n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
        has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
        monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
        is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
        missing_values_bin_idx = n_bins - 1
        splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
        assert np.all(sample_indices == splitter.partition)
        histograms = builder.compute_histograms_brute(sample_indices)
        value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
        si_root = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, allowed_features=None)
        assert si_root.feature_idx == 1
        si_root = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, allowed_features=allowed_features)
        split_features.append(si_root.feature_idx)
        assert si_root.feature_idx in allowed_features
    assert set(allowed_features) == set(split_features)