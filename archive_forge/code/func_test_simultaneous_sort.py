import itertools
import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.utils import check_random_state
@pytest.mark.parametrize('simultaneous_sort', [simultaneous_sort_bt, simultaneous_sort_kdt])
def test_simultaneous_sort(simultaneous_sort, n_rows=10, n_pts=201):
    rng = check_random_state(0)
    dist = rng.random_sample((n_rows, n_pts)).astype(np.float64, copy=False)
    ind = (np.arange(n_pts) + np.zeros((n_rows, 1))).astype(np.intp, copy=False)
    dist2 = dist.copy()
    ind2 = ind.copy()
    simultaneous_sort(dist, ind)
    i = np.argsort(dist2, axis=1)
    row_ind = np.arange(n_rows)[:, None]
    dist2 = dist2[row_ind, i]
    ind2 = ind2[row_ind, i]
    assert_array_almost_equal(dist, dist2)
    assert_array_almost_equal(ind, ind2)