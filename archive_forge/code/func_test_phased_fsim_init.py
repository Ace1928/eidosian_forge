import numpy as np
import pytest
import sympy
import cirq
def test_phased_fsim_init():
    f = cirq.PhasedFSimGate(1, 2, 3, 4, 5)
    assert f.theta == 1
    assert f.zeta == 2
    assert f.chi == 3
    assert f.gamma == 4 - 2 * np.pi
    assert f.phi == 5 - 2 * np.pi
    f2 = cirq.PhasedFSimGate(theta=1, zeta=2, chi=3, gamma=4, phi=5)
    assert f2.theta == 1
    assert f2.zeta == 2
    assert f2.chi == 3
    assert f2.gamma == 4 - 2 * np.pi
    assert f2.phi == 5 - 2 * np.pi