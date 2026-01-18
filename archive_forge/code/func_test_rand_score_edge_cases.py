import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
@pytest.mark.parametrize('clustering1, clustering2', [(list(range(100)), list(range(100))), (np.zeros((100,)), np.zeros((100,)))])
def test_rand_score_edge_cases(clustering1, clustering2):
    assert_allclose(rand_score(clustering1, clustering2), 1.0)