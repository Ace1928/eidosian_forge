import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_gates_known_x():
    actual = cirq.single_qubit_matrix_to_gates(np.array([[0, 1], [1, 0]]), tolerance=0.01)
    assert cirq.approx_eq(actual, [cirq.X], atol=1e-09)