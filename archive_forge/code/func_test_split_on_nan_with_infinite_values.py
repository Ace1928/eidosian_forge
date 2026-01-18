import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_split_on_nan_with_infinite_values():
    X = np.array([0, 1, np.inf, np.nan, np.nan]).reshape(-1, 1)
    gradients = np.array([0, 0, 0, 100, 100], dtype=G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)
    n_bins_non_missing = 3
    has_missing_values = True
    grower = TreeGrower(X_binned, gradients, hessians, n_bins_non_missing=n_bins_non_missing, has_missing_values=has_missing_values, min_samples_leaf=1, n_threads=n_threads)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=bin_mapper.bin_thresholds_)
    assert predictor.nodes[0]['num_threshold'] == np.inf
    assert predictor.nodes[0]['bin_threshold'] == n_bins_non_missing - 1
    known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()
    predictions = predictor.predict(X, known_cat_bitsets, f_idx_map, n_threads)
    predictions_binned = predictor.predict_binned(X_binned, missing_values_bin_idx=bin_mapper.missing_values_bin_idx_, n_threads=n_threads)
    np.testing.assert_allclose(predictions, -gradients)
    np.testing.assert_allclose(predictions_binned, -gradients)