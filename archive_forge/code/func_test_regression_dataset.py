import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.ensemble._hist_gradient_boosting._bitset import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('n_bins', [200, 256])
def test_regression_dataset(n_bins):
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    mapper = _BinMapper(n_bins=n_bins, random_state=42)
    X_train_binned = mapper.fit_transform(X_train)
    gradients = -y_train.astype(G_H_DTYPE)
    hessians = np.ones(1, dtype=G_H_DTYPE)
    min_samples_leaf = 10
    max_leaf_nodes = 30
    grower = TreeGrower(X_train_binned, gradients, hessians, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, n_bins=n_bins, n_bins_non_missing=mapper.n_bins_non_missing_)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)
    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)
    y_pred_train = predictor.predict(X_train, known_cat_bitsets, f_idx_map, n_threads)
    assert r2_score(y_train, y_pred_train) > 0.82
    y_pred_test = predictor.predict(X_test, known_cat_bitsets, f_idx_map, n_threads)
    assert r2_score(y_test, y_pred_test) > 0.67