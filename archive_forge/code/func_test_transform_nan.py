import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_transform_nan():
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)
    Y[:, 0] = 0
    estimator = SparsePCA(n_components=8)
    assert not np.any(np.isnan(estimator.fit_transform(Y)))