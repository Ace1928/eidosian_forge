import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_rotations(rng):
    for i in range(TEST_N):
        rotm = rand_rotm(rng)
        q = Quaternion.from_matrix(rotm)
        assert np.allclose(rotm, q.rotation_matrix())
        v = rng.rand(3)
        vrotM = np.dot(rotm, v)
        vrotQ = q.rotate(v)
        assert np.allclose(vrotM, vrotQ)