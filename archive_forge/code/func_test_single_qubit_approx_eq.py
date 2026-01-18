import re
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_approx_eq():
    x = cirq.MatrixGate(np.array([[0, 1], [1, 0]]))
    i = cirq.MatrixGate(np.array([[1, 0], [0, 1]]))
    i_ish = cirq.MatrixGate(np.array([[1, 1e-15], [0, 1]]))
    assert cirq.approx_eq(i, i_ish, atol=1e-09)
    assert cirq.approx_eq(i, i, atol=1e-09)
    assert not cirq.approx_eq(i, x, atol=1e-09)
    assert not cirq.approx_eq(i, '', atol=1e-09)