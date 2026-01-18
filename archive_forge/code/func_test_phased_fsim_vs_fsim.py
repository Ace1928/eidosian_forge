import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, phi', ((0, 0), (0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1), (np.pi / 2, np.pi / 6), (np.pi, np.pi), (3.5 * np.pi, 4 * np.pi)))
def test_phased_fsim_vs_fsim(theta, phi):
    g1 = cirq.FSimGate(theta, phi)
    g2 = cirq.PhasedFSimGate(theta, 0, 0, 0, phi)
    assert np.allclose(cirq.unitary(g1), cirq.unitary(g2))