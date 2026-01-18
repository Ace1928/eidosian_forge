import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_call_time_dim_mismatch():
    rnd = np.random.RandomState(0)
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    t = np.arange(5)
    s = Slerp(t, r)
    with pytest.raises(ValueError, match='`times` must be at most 1-dimensional.'):
        interp_times = np.array([[3.5], [4.2]])
        s(interp_times)