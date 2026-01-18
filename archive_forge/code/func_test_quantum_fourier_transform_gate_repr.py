import numpy as np
import pytest, sympy
import cirq
def test_quantum_fourier_transform_gate_repr():
    b = cirq.QuantumFourierTransformGate(num_qubits=2, without_reverse=False)
    cirq.testing.assert_equivalent_repr(b)