import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_random_rotation_shape():
    rnd = np.random.RandomState(0)
    assert_equal(Rotation.random(random_state=rnd).as_quat().shape, (4,))
    assert_equal(Rotation.random(None, random_state=rnd).as_quat().shape, (4,))
    assert_equal(Rotation.random(1, random_state=rnd).as_quat().shape, (1, 4))
    assert_equal(Rotation.random(5, random_state=rnd).as_quat().shape, (5, 4))