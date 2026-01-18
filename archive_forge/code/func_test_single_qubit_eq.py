import re
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MatrixGate(np.eye(2)))
    eq.make_equality_group(lambda: cirq.MatrixGate(np.array([[0, 1], [1, 0]])))
    x2 = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    eq.make_equality_group(lambda: cirq.MatrixGate(x2))
    eq.add_equality_group(cirq.MatrixGate(PLUS_ONE, qid_shape=(3,)))