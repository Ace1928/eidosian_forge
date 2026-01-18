import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_phased_iswap_pow():
    gate1 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.25)
    gate2 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.5)
    assert gate1 ** 2 == gate2
    u1 = cirq.unitary(gate1)
    u2 = cirq.unitary(gate2)
    assert np.allclose(u1 @ u1, u2)
    gate1 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.25, global_shift=0.25)
    gate2 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.5, global_shift=0.25)
    assert gate1 ** 2 == gate2
    u1 = cirq.unitary(gate1)
    u2 = cirq.unitary(gate2)
    assert np.allclose(u1 @ u1, u2)