import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('rz_angles_before, rz_angles_after', (((0, 0), (0, 0)), ((0.1, 0.2), (0.3, 0.7)), ((-0.1, 0.2), (0.3, 0.8)), ((-0.1, -0.2), (0.3, -0.9)), ((np.pi, np.pi / 6), (-np.pi / 2, 0))))
def test_phased_fsim_phase_angle_symmetry(rz_angles_before, rz_angles_after):
    f = cirq.PhasedFSimGate.from_fsim_rz(np.pi / 3, np.pi / 5, rz_angles_before, rz_angles_after)
    for d in (-10, -7, -2 * np.pi, -0.2, 0, 0.1, 0.2, np.pi, 8, 20):
        rz_angles_before2 = (rz_angles_before[0] + d, rz_angles_before[1] + d)
        rz_angles_after2 = (rz_angles_after[0] - d, rz_angles_after[1] - d)
        f2 = cirq.PhasedFSimGate.from_fsim_rz(np.pi / 3, np.pi / 5, rz_angles_before2, rz_angles_after2)
        assert cirq.approx_eq(f, f2)