import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pauli_sum_exponential_parameterized_matrix_raises():
    with pytest.raises(ValueError, match='parameterized'):
        cirq.PauliSumExponential(cirq.X(q0) + cirq.Z(q1), sympy.Symbol('theta')).matrix()