import numpy as np
import pytest, sympy
import cirq
def test_phase_gradient():
    np.testing.assert_allclose(cirq.unitary(cirq.PhaseGradientGate(num_qubits=2, exponent=1)), np.diag([1, 1j, -1, -1j]))
    for k in range(4):
        cirq.testing.assert_implements_consistent_protocols(cirq.PhaseGradientGate(num_qubits=k, exponent=1))