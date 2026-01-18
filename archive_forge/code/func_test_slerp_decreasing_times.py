import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_decreasing_times():
    with pytest.raises(ValueError, match='strictly increasing order'):
        rnd = np.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        t = [0, 1, 3, 2, 4]
        Slerp(t, r)