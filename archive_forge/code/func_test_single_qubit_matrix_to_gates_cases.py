import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('intended_effect', [np.array([[0, 1j], [1, 0]]), np.array([[-0.10313355 - 0.62283483j, 0.76512225 - 0.1266025j], [-0.72184177 + 0.28352196j, 0.23073193 + 0.5876415j]])] + [cirq.testing.random_unitary(2) for _ in range(10)])
def test_single_qubit_matrix_to_gates_cases(intended_effect):
    for atol in [0.1, 1e-08]:
        gates = cirq.single_qubit_matrix_to_gates(intended_effect, tolerance=atol / 10)
        assert len(gates) <= 3
        assert sum((1 for g in gates if not isinstance(g, cirq.ZPowGate))) <= 1
        assert_gates_implement_unitary(gates, intended_effect, atol=atol)