import re
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_init():
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    x2 = cirq.MatrixGate(m)
    assert cirq.has_unitary(x2)
    assert np.all(cirq.unitary(x2) == m)
    assert cirq.qid_shape(x2) == (2,)
    x2 = cirq.MatrixGate(PLUS_ONE, qid_shape=(3,))
    assert cirq.has_unitary(x2)
    assert np.all(cirq.unitary(x2) == PLUS_ONE)
    assert cirq.qid_shape(x2) == (3,)
    with pytest.raises(ValueError, match='Not a .*unitary matrix'):
        cirq.MatrixGate(np.zeros((2, 2)))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(cirq.eye_tensor((2, 2), dtype=float))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(np.ones((3, 4)))
    with pytest.raises(ValueError, match='must be a square 2d numpy array'):
        cirq.MatrixGate(np.ones((2, 2, 2)))