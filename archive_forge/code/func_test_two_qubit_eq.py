import re
import numpy as np
import pytest
import sympy
import cirq
def test_two_qubit_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.MatrixGate(np.eye(4)))
    eq.make_equality_group(lambda: cirq.MatrixGate(QFT2))
    eq.make_equality_group(lambda: cirq.MatrixGate(HH))