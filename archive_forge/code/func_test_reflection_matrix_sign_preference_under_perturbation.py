import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_reflection_matrix_sign_preference_under_perturbation():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = np.array([[1, -1j], [-1j, 1]]) * (1 + 1j) / 2
    np.testing.assert_allclose(cirq.reflection_matrix_pow(x, 0.5), sqrt_x, atol=1e-08)
    for perturbation in [0, 0.1, -0.1, 0.3, -0.3, 0.49, -0.49]:
        px = x * complex(-1) ** perturbation
        expected_sqrt_px = sqrt_x * complex(-1) ** (perturbation / 2)
        sqrt_px = cirq.reflection_matrix_pow(px, 0.5)
        np.testing.assert_allclose(np.dot(sqrt_px, sqrt_px), px, atol=1e-10)
        np.testing.assert_allclose(sqrt_px, expected_sqrt_px, atol=1e-10)