import numpy as np
import pytest
import sympy
import cirq
def test_xx_matrix():
    np.testing.assert_allclose(cirq.unitary(cirq.XX), np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.XX ** 2), np.eye(4), atol=1e-08)
    c = np.cos(np.pi / 6)
    s = -1j * np.sin(np.pi / 6)
    np.testing.assert_allclose(cirq.unitary(cirq.XXPowGate(exponent=1 / 3, global_shift=-0.5)), np.array([[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]]), atol=1e-08)