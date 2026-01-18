import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_overload(rng):
    for i in range(TEST_N):
        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)
        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)
        assert np.allclose(np.dot(rotm2, rotm1), (q2 * q1).rotation_matrix())
        v = rng.rand(3)
        vrotM = np.dot(rotm2, np.dot(rotm1, v))
        vrotQ = (q2 * q1).rotate(v)
        assert np.allclose(vrotM, vrotQ)