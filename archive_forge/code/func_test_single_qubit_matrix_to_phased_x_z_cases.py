import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('intended_effect', [np.array([[0, 1j], [1, 0]])] + [cirq.testing.random_unitary(2) for _ in range(10)])
def test_single_qubit_matrix_to_phased_x_z_cases(intended_effect):
    gates = cirq.single_qubit_matrix_to_phased_x_z(intended_effect, atol=1e-06)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-05)