import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('intended_effect', [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, 1j]]), np.array([[1, 0], [0, -1j]]), np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), np.array([[0, 1j], [1, 0]]), *[cirq.testing.random_unitary(2) for _ in range(10)]])
def test_single_qubit_matrix_to_phxz_cases(intended_effect):
    gate = cirq.single_qubit_matrix_to_phxz(intended_effect, atol=1e-06)
    assert_gates_implement_unitary([gate], intended_effect, atol=1e-05)