import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_setitem_integer():
    rng = np.random.RandomState(seed=0)
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(random_state=rng)
    r1[1] = r2
    assert_equal(r1[1].as_quat(), r2.as_quat())