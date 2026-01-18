import pytest
import numpy as np
from ase.quaternions import Quaternion
def test_quaternions_euler(rng):
    for mode in ['zyz', 'zxz']:
        for i in range(TEST_N):
            abc = rng.rand(3) * 2 * np.pi
            q_eul = Quaternion.from_euler_angles(*abc, mode=mode)
            rot_eul = eulang_rotm(*abc, mode=mode)
            assert np.allclose(rot_eul, q_eul.rotation_matrix())
            abc_2 = q_eul.euler_angles(mode=mode)
            q_eul_2 = Quaternion.from_euler_angles(*abc_2, mode=mode)
            assert np.allclose(q_eul_2.q, q_eul.q)