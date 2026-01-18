import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_time_dim_mismatch():
    with pytest.raises(ValueError, match='times to be specified in a 1 dimensional array'):
        rnd = np.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(2, 4)))
        t = np.array([[1], [2]])
        Slerp(t, r)