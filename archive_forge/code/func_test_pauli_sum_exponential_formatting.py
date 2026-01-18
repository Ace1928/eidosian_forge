import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('psum_exp, expected_str', ((cirq.PauliSumExponential(0, np.pi / 2), 'exp(j * 1.5707963267948966 * (0.000))'), (cirq.PauliSumExponential(2j * cirq.X(q0) + 4j * cirq.Y(q1), 2), 'exp(2 * (2.000j*X(q(0))+4.000j*Y(q(1))))'), (cirq.PauliSumExponential(0.5 * cirq.X(q0) + 0.6 * cirq.Y(q1), sympy.Symbol('theta')), 'exp(j * theta * (0.500*X(q(0))+0.600*Y(q(1))))')))
def test_pauli_sum_exponential_formatting(psum_exp, expected_str):
    assert str(psum_exp) == expected_str