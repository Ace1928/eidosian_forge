import numpy as np
import pytest
import sympy
import cirq
def test_zz_matrix():
    np.testing.assert_allclose(cirq.unitary(cirq.ZZ), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.ZZ ** 2), np.eye(4), atol=1e-08)
    b = 1j ** 0.25
    a = np.conj(b)
    np.testing.assert_allclose(cirq.unitary(cirq.ZZPowGate(exponent=0.25, global_shift=-0.5)), np.array([[a, 0, 0, 0], [0, b, 0, 0], [0, 0, b, 0], [0, 0, 0, a]]), atol=1e-08)