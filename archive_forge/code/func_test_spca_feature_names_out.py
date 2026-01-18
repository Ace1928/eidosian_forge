import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('SPCA', [SparsePCA, MiniBatchSparsePCA])
def test_spca_feature_names_out(SPCA):
    """Check feature names out for *SparsePCA."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (12, 10)
    X = rng.randn(n_samples, n_features)
    model = SPCA(n_components=4).fit(X)
    names = model.get_feature_names_out()
    estimator_name = SPCA.__name__.lower()
    assert_array_equal([f'{estimator_name}{i}' for i in range(4)], names)