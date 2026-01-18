import re
import numpy as np
import pytest
import sympy
import cirq
def test_two_qubit_init():
    x2 = cirq.MatrixGate(QFT2)
    assert cirq.has_unitary(x2)
    assert np.all(cirq.unitary(x2) == QFT2)