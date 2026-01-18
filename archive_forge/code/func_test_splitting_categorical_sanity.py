import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
@pytest.mark.parametrize('X_binned, all_gradients, expected_categories_left, n_bins_non_missing,missing_values_bin_idx, has_missing_values, expected_missing_go_to_left', [([0, 1, 2, 3] * 11, [10, 1, 10, 10] * 11, [1], 4, 4, False, None), ([0, 1, 2, 3] * 11, [10, 10, 10, 1] * 11, [3], 4, 4, False, None), ([0, 1, 2, 3] * 11 + [4] * 5, [10, 10, 10, 1] * 11 + [10] * 5, [3], 4, 4, False, None), ([0, 1, 2, 3] * 11 + [4] * 5, [10, 10, 10, 1] * 11 + [1] * 5, [3], 4, 4, False, None), ([0, 1, 2] * 11 + [9] * 11, [10, 1, 10] * 11 + [10] * 11, [1], 3, 9, True, False), ([0, 1, 2] * 11 + [9] * 11, [10, 1, 10] * 11 + [1] * 11, [1, 9], 3, 9, True, True), ([0, 1, 2, 3, 4] * 11 + [255] * 12, [10, 10, 10, 10, 10] * 11 + [1] * 12, [255], 5, 255, True, True), (list(range(60)) * 12, [10, 1] * 360, list(range(1, 60, 2)), 59, 59, True, True), (list(range(256)) * 12, [10, 10, 10, 10, 10, 10, 10, 1] * 384, list(range(7, 256, 8)), 255, 255, True, True)])
def test_splitting_categorical_sanity(X_binned, all_gradients, expected_categories_left, n_bins_non_missing, missing_values_bin_idx, has_missing_values, expected_missing_go_to_left):
    n_samples = len(X_binned)
    n_bins = max(X_binned) + 1
    X_binned = np.array(X_binned, dtype=X_BINNED_DTYPE).reshape(-1, 1)
    X_binned = np.asfortranarray(X_binned)
    l2_regularization = 0.0
    min_hessian_to_split = 0.001
    min_samples_leaf = 1
    min_gain_to_split = 0.0
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    all_gradients = np.array(all_gradients, dtype=G_H_DTYPE)
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    has_missing_values = np.array([has_missing_values], dtype=np.uint8)
    sum_gradients = all_gradients.sum()
    sum_hessians = n_samples
    hessians_are_constant = True
    builder = HistogramBuilder(X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads)
    n_bins_non_missing = np.array([n_bins_non_missing], dtype=np.uint32)
    monotonic_cst = np.array([MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8)
    is_categorical = np.ones_like(monotonic_cst, dtype=np.uint8)
    splitter = Splitter(X_binned, n_bins_non_missing, missing_values_bin_idx, has_missing_values, is_categorical, monotonic_cst, l2_regularization, min_hessian_to_split, min_samples_leaf, min_gain_to_split, hessians_are_constant)
    histograms = builder.compute_histograms_brute(sample_indices)
    value = compute_node_value(sum_gradients, sum_hessians, -np.inf, np.inf, l2_regularization)
    split_info = splitter.find_node_split(n_samples, histograms, sum_gradients, sum_hessians, value)
    assert split_info.is_categorical
    assert split_info.gain > 0
    _assert_categories_equals_bitset(expected_categories_left, split_info.left_cat_bitset)
    if has_missing_values:
        assert split_info.missing_go_to_left == expected_missing_go_to_left
    samples_left, samples_right, _ = splitter.split_indices(split_info, splitter.partition)
    left_mask = np.isin(X_binned.ravel(), expected_categories_left)
    assert_array_equal(sample_indices[left_mask], samples_left)
    assert_array_equal(sample_indices[~left_mask], samples_right)