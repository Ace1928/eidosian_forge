import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('theta, zeta, chi, gamma, phi', [(0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (np.pi / 3, 0, 0, 0, np.pi / 5), (np.pi / 3, 1, 0, 0, np.pi / 5), (np.pi / 3, 0, 1, 0, np.pi / 5), (np.pi / 3, 0, 0, 1, np.pi / 5), (-np.pi / 3, 1, 0, 0, np.pi / 5), (np.pi / 3, 0, 1, 0, -np.pi / 5), (-np.pi / 3, 0, 0, 1, -np.pi / 5), (np.pi, 0, 0, sympy.Symbol('a'), 0)])
def test_phased_fsim_consistent(theta, zeta, chi, gamma, phi):
    gate = cirq.PhasedFSimGate(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi)
    cirq.testing.assert_implements_consistent_protocols(gate)