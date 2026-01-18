import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp, expected_psum_exp', ((cirq.PauliSumExponential(cirq.Z(q0), np.pi / 2), cirq.PauliSumExponential(cirq.Z(q1), np.pi / 2)), (cirq.PauliSumExponential(2j * cirq.X(q0) + 3j * cirq.Y(q2), sympy.Symbol('theta')), cirq.PauliSumExponential(2j * cirq.X(q1) + 3j * cirq.Y(q3), sympy.Symbol('theta'))), (cirq.PauliSumExponential(cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), np.pi), cirq.PauliSumExponential(cirq.X(q1) * cirq.Y(q2) + cirq.Y(q2) * cirq.Z(q3), np.pi))))
def test_pauli_sum_exponential_with_qubits(psum_exp, expected_psum_exp):
    assert psum_exp.with_qubits(*expected_psum_exp.qubits) == expected_psum_exp