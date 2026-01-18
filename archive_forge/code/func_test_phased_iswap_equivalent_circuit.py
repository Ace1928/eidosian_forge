import numpy as np
import pytest
import scipy
import sympy
import cirq
def test_phased_iswap_equivalent_circuit():
    p = 0.7
    t = -0.4
    gate = cirq.PhasedISwapPowGate(phase_exponent=p, exponent=t)
    q0, q1 = cirq.LineQubit.range(2)
    equivalent_circuit = cirq.Circuit([cirq.Z(q0) ** p, cirq.Z(q1) ** (-p), cirq.ISWAP(q0, q1) ** t, cirq.Z(q0) ** (-p), cirq.Z(q1) ** p])
    assert np.allclose(cirq.unitary(gate), cirq.unitary(equivalent_circuit))