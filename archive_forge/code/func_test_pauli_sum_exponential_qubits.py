import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp, expected_qubits', ((cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2), (q1,)), (cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol('theta')), (q0, q2)), (cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q2) * cirq.Z(q3), np.pi), (q0, q1, q2, q3))))
def test_pauli_sum_exponential_qubits(psum_exp, expected_qubits):
    assert psum_exp.qubits == expected_qubits