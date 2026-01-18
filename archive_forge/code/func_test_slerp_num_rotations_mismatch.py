import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_num_rotations_mismatch():
    with pytest.raises(ValueError, match='number of rotations to be equal to number of timestamps'):
        rnd = np.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        t = np.arange(7)
        Slerp(t, r)