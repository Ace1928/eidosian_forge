import numpy as np
import pytest
import scipy
import sympy
import cirq
@pytest.mark.parametrize('phase_exponent, exponent, global_shift', [(0, 0, 0), (0, 0.1, 0.1), (0, 0.5, 0.5), (0, -1, 0.2), (-0.3, 0, 0.3), (0.1, 0.1, 0.6), (0.1, 0.5, 0.7), (0.5, 0.5, 0.8), (-0.1, 0.1, 0.9), (-0.5, 1, 1), (0.3, 2, 0.1), (0.4, -2, 0.25), (0.1, sympy.Symbol('p'), 0.33), (sympy.Symbol('t'), 0.5, 0.86), (sympy.Symbol('t'), sympy.Symbol('p'), 1)])
def test_phased_iswap_has_consistent_protocols(phase_exponent, exponent, global_shift):
    cirq.testing.assert_implements_consistent_protocols(cirq.PhasedISwapPowGate(phase_exponent=phase_exponent, exponent=exponent, global_shift=global_shift))