import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_reflection_matrix_pow_consistent_results():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = cirq.reflection_matrix_pow(x, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_x, sqrt_x), x, atol=1e-10)
    ix = x * np.sqrt(1j)
    sqrt_ix = cirq.reflection_matrix_pow(ix, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_ix, sqrt_ix), ix, atol=1e-10)
    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    cube_root_h = cirq.reflection_matrix_pow(h, 1 / 3)
    np.testing.assert_allclose(np.dot(np.dot(cube_root_h, cube_root_h), cube_root_h), h, atol=1e-08)
    y = np.array([[0, -1j], [1j, 0]])
    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5j)
    yh = np.kron(y, h)
    sqrt_yh = cirq.reflection_matrix_pow(yh, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_yh, sqrt_yh), yh, atol=1e-10)