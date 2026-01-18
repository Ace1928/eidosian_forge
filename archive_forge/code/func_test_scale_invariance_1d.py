import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
@pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
def test_scale_invariance_1d(self, kernel):
    seq = Halton(1, scramble=False, seed=np.random.RandomState())
    x = 3 * seq.random(50)
    y = _1d_test_function(x)
    xitp = 3 * seq.random(50)
    yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
    yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
    assert_allclose(yitp1, yitp2, atol=1e-08)