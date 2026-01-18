import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_weighted_mean():
    axes = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    thetas = np.linspace(0, np.pi / 2, 100)
    for t in thetas:
        rw = Rotation.from_rotvec(t * axes[:2])
        mw = rw.mean(weights=[1, 2])
        r = Rotation.from_rotvec(t * axes)
        m = r.mean()
        assert_allclose((m * mw.inv()).magnitude(), 0, atol=1e-10)