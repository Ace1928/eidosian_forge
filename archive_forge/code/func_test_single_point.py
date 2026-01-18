import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_single_point(self):
    for dim in [1, 2, 3]:
        y = np.zeros((1, dim))
        d = np.ones((1,))
        f = self.build(y, d, kernel='linear')(y)
        assert_allclose(d, f)