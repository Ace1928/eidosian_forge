import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('terms, expected_unitary', (({cirq.X: np.sqrt(0.5), cirq.Y: np.sqrt(0.5)}, np.array([[0, np.sqrt(-1j)], [np.sqrt(1j), 0]])), ({cirq.IdentityGate(2): np.sqrt(0.5), cirq.YY: -1j * np.sqrt(0.5)}, np.sqrt(0.5) * np.array([[1, 0, 0, 1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [1j, 0, 0, 1]]))))
def test_unitary_linear_combination_of_gates_has_correct_unitary(terms, expected_unitary):
    combination = cirq.LinearCombinationOfGates(terms)
    assert cirq.has_unitary(combination)
    assert np.allclose(cirq.unitary(combination), expected_unitary)