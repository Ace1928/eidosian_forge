import re
import numpy as np
import pytest
import sympy
import cirq
def test_two_qubit_approx_eq():
    f = cirq.MatrixGate(QFT2)
    perturb = np.zeros(shape=QFT2.shape, dtype=np.float64)
    perturb[1, 2] = 1e-08
    assert cirq.approx_eq(f, cirq.MatrixGate(QFT2), atol=1e-09)
    assert not cirq.approx_eq(f, cirq.MatrixGate(QFT2 + perturb), atol=1e-09)
    assert cirq.approx_eq(f, cirq.MatrixGate(QFT2 + perturb), atol=1e-07)
    assert not cirq.approx_eq(f, cirq.MatrixGate(HH), atol=1e-09)