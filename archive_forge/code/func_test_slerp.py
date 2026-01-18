import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_slerp():
    rnd = np.random.RandomState(0)
    key_rots = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    key_quats = key_rots.as_quat()
    key_times = [0, 1, 2, 3, 4]
    interpolator = Slerp(key_times, key_rots)
    times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.6, 4]
    interp_rots = interpolator(times)
    interp_quats = interp_rots.as_quat()
    interp_quats[interp_quats[:, -1] < 0] *= -1
    key_quats[key_quats[:, -1] < 0] *= -1
    assert_allclose(interp_quats[0], key_quats[0])
    assert_allclose(interp_quats[3], key_quats[1])
    assert_allclose(interp_quats[5], key_quats[2])
    assert_allclose(interp_quats[7], key_quats[3])
    assert_allclose(interp_quats[10], key_quats[4])
    cos_theta1 = np.sum(interp_quats[0] * interp_quats[2])
    cos_theta2 = np.sum(interp_quats[2] * interp_quats[1])
    assert_allclose(cos_theta1, cos_theta2)
    cos_theta4 = np.sum(interp_quats[3] * interp_quats[4])
    cos_theta5 = np.sum(interp_quats[4] * interp_quats[5])
    assert_allclose(cos_theta4, cos_theta5)
    cos_theta3 = np.sum(interp_quats[1] * interp_quats[3])
    assert_allclose(cos_theta3, 2 * cos_theta1 ** 2 - 1)
    assert_equal(len(interp_rots), len(times))