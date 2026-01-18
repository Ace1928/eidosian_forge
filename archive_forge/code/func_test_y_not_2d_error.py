import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_y_not_2d_error(self):
    y = np.linspace(0, 1, 5)
    d = np.zeros(5)
    match = '`y` must be a 2-dimensional array.'
    with pytest.raises(ValueError, match=match):
        self.build(y, d)