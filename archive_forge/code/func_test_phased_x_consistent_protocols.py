import itertools
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('phase_exponent', [-0.5, 0, 0.5, 1, sympy.Symbol('p'), sympy.Symbol('p') + 1])
def test_phased_x_consistent_protocols(phase_exponent):
    cirq.testing.assert_implements_consistent_protocols(cirq.PhasedXPowGate(phase_exponent=phase_exponent, exponent=1.0))
    cirq.testing.assert_implements_consistent_protocols(cirq.PhasedXPowGate(phase_exponent=phase_exponent, exponent=1.0, global_shift=0.1))