import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@pytest.mark.parametrize('forbidden_features', [set(), {1, 3}])
def test_split_feature_fraction_per_split(forbidden_features):
    """Check that feature_fraction_per_split is respected.

    Because we set `n_features = 4` and `feature_fraction_per_split = 0.25`, it means
    that calling `splitter.find_node_split` will be allowed to select a split for a
    single completely random feature at each call. So if we iterate enough, we should
    cover all the allowed features, irrespective of the values of the gradients and
    Hessians of the objective.
    """
    n_features = 4
    allowed_features = np.array(list(set(range(n_features)) - forbidden_features), dtype=np.uint32)
    n_bins = 5
    n_samples = 40
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    rng = np.random.default_rng(42)
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = rng.uniform(low=0.5, high=1, size=n_samples).astype(G_H_DTYPE)
    sum_gradients = all_gradients.sum()
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    sum_hessians = n_samples
    hessians_are_constant = True
    X_binned = np.asfortranarray(rng.integers(low=0, high=n_bins - 1, size=(n_samples, n_features)), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned, dtype=X_BINNED_DTYPE)
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    n_bins_non_missing = np.array([n_bins] * X_binned.shape[1], dtype=np.uint32)
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    missing_values_bin_idx = n_bins - 1
    params = dict(X_binned=X_binned, n_bins_non_missing=n_bins_non_missing, missing_values_bin_idx=missing_values_bin_idx, has_missing_values=has_missing_values, is_categorical=is_categorical, monotonic_cst=monotonic_cst, l2_regularization=l2_regularization, min_hessian_to_split=min_hessian_to_split, min_samples_leaf=min_samples_leaf, min_gain_to_split=min_gain_to_split, hessians_are_constant=hessians_are_constant, rng=rng)
    splitter_subsample = Splitter(feature_fraction_per_split=0.25, **params)
    splitter_all_features = Splitter(feature_fraction_per_split=1.0, **params)
    assert np.all(sample_indices == splitter_subsample.partition)
    split_features_subsample = []
    split_features_all = []
    for i in range(20):
        si_root = splitter_subsample.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, allowed_features=allowed_features)
        split_features_subsample.append(si_root.feature_idx)
        si_root = splitter_all_features.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value, allowed_features=allowed_features)
        split_features_all.append(si_root.feature_idx)
    assert set(split_features_subsample) == set(allowed_features)
    assert len(set(split_features_all)) == 1