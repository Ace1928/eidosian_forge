import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_axang(rng):
    q = Quaternion()
    n, theta = q.axis_angle()
    assert theta == 0
    u = np.array([1, 0.5, 1])
    u /= np.linalg.norm(u)
    alpha = 1.25
    q = Quaternion.from_matrix(axang_rotm(u, alpha))
    n, theta = q.axis_angle()
    assert np.isclose(theta, alpha)
    assert np.allclose(u, n)