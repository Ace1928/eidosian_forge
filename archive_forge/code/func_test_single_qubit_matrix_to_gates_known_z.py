import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_gates_known_z():
    actual = cirq.single_qubit_matrix_to_gates(np.array([[1, 0], [0, -1]]), tolerance=0.01)
    assert cirq.approx_eq(actual, [cirq.Z], atol=1e-09)