import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_left_multiply_reorders_matrices():
    t = np.eye(4).reshape((2, 2, 2, 2))
    m = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]).reshape((2, 2, 2, 2))
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t, target_axes=[0, 1]), m, atol=1e-08)
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t, target_axes=[1, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]).reshape((2, 2, 2, 2)), atol=1e-08)