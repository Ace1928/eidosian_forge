import numpy as np
import pytest
import sympy
import cirq
def test_protocols_mul_not_implemented():
    diagonal_angles = [2, 3, None, 7]
    diagonal_gate = cirq.TwoQubitDiagonalGate(diagonal_angles)
    with pytest.raises(TypeError):
        cirq.protocols.pow(diagonal_gate, 3)