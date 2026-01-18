import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('SPCA', (SparsePCA, MiniBatchSparsePCA))
@pytest.mark.parametrize('method', ('lars', 'cd'))
@pytest.mark.parametrize('data_type, expected_type', ((np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)))
def test_sparse_pca_dtype_match(SPCA, method, data_type, expected_type):
    n_samples, n_features, n_components = (12, 10, 3)
    rng = np.random.RandomState(0)
    input_array = rng.randn(n_samples, n_features).astype(data_type)
    model = SPCA(n_components=n_components, method=method)
    transformed = model.fit_transform(input_array)
    assert transformed.dtype == expected_type
    assert model.components_.dtype == expected_type