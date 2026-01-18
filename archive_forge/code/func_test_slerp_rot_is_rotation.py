import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp_rot_is_rotation():
    with pytest.raises(TypeError, match='must be a `Rotation` instance'):
        r = np.array([[1, 2, 3, 4], [0, 0, 0, 1]])
        t = np.array([0, 1])
        Slerp(t, r)