import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_rotm(rng):
    for i in range(TEST_N):
        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)
        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)
        assert np.allclose(q1.rotation_matrix(), rotm1)
        assert np.allclose(q2.rotation_matrix(), rotm2)
        assert np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2))
        assert np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2))