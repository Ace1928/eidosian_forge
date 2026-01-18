import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_unique_categories', (2, 10, 100))
@pytest.mark.parametrize('target', ('binary', 'random', 'equal'))
def test_ohe_equivalence(min_samples_leaf, n_unique_categories, target):
    rng = np.random.RandomState(0)
    n_samples = 10000
    X_binned = rng.randint(0, n_unique_categories, size=(n_samples, 1), dtype=np.uint8)
    X_ohe = OneHotEncoder(sparse_output=False).fit_transform(X_binned)
    X_ohe = np.asfortranarray(X_ohe).astype(np.uint8)
    if target == 'equal':
        gradients = X_binned.reshape(-1)
    elif target == 'binary':
        gradients = (X_binned % 2).reshape(-1)
    else:
        gradients = rng.randn(n_samples)
    gradients = gradients.astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    grower_params = {'min_samples_leaf': min_samples_leaf, 'max_depth': None, 'max_leaf_nodes': None}
    grower = TreeGrower(X_binned, gradients, hessians, is_categorical=[True], **grower_params)
    grower.grow()
    predictor = grower.make_predictor(binning_thresholds=np.zeros((1, n_unique_categories)))
    preds = predictor.predict_binned(X_binned, missing_values_bin_idx=255, n_threads=n_threads)
    grower_ohe = TreeGrower(X_ohe, gradients, hessians, **grower_params)
    grower_ohe.grow()
    predictor_ohe = grower_ohe.make_predictor(binning_thresholds=np.zeros((X_ohe.shape[1], n_unique_categories)))
    preds_ohe = predictor_ohe.predict_binned(X_ohe, missing_values_bin_idx=255, n_threads=n_threads)
    assert predictor.get_max_depth() <= predictor_ohe.get_max_depth()
    if target == 'binary' and n_unique_categories > 2:
        assert predictor.get_max_depth() < predictor_ohe.get_max_depth()
    np.testing.assert_allclose(preds, preds_ohe)