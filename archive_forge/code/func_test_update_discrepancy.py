import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_update_discrepancy(self):
    space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
    disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
    disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)
    assert_allclose(disc_iter, 0.0081, atol=0.0001)
    rng = np.random.default_rng(241557431858162136881731220526394276199)
    space_1 = rng.random((4, 10))
    disc_ref = qmc.discrepancy(space_1)
    disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
    disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)
    assert_allclose(disc_iter, disc_ref, atol=0.0001)
    with pytest.raises(ValueError, match='Sample is not in unit hypercube'):
        update_discrepancy(space_1[-1], space_1[:-1] + 1, disc_init)
    with pytest.raises(ValueError, match='Sample is not a 2D array'):
        update_discrepancy(space_1[-1], space_1[0], disc_init)
    x_new = [1, 3]
    with pytest.raises(ValueError, match='x_new is not in unit hypercube'):
        update_discrepancy(x_new, space_1[:-1], disc_init)
    x_new = [[0.5, 0.5]]
    with pytest.raises(ValueError, match='x_new is not a 1D array'):
        update_discrepancy(x_new, space_1[:-1], disc_init)
    x_new = [0.3, 0.1, 0]
    with pytest.raises(ValueError, match='x_new and sample must be broadcastable'):
        update_discrepancy(x_new, space_1[:-1], disc_init)