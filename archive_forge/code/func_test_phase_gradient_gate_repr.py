import numpy as np
import pytest, sympy
import cirq
def test_phase_gradient_gate_repr():
    a = cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)
    cirq.testing.assert_equivalent_repr(a)