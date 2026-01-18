import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_left_multiply_matches_kron_then_dot():
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = np.array([[2, 3], [5, 7]])
    i = np.eye(2)
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[0]), np.dot(cirq.kron(m, i, i), t).reshape((2, 2, 2)), atol=1e-08)
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[1]), np.dot(cirq.kron(i, m, i), t).reshape((2, 2, 2)), atol=1e-08)
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[2]), np.dot(cirq.kron(i, i, m), t).reshape((2, 2, 2)), atol=1e-08)
    np.testing.assert_allclose(cirq.targeted_left_multiply(left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[2]), np.dot(cirq.kron(i, i, m), t).reshape((2, 2, 2)), atol=1e-08)
    common = t.reshape((2, 2, 2))
    with pytest.raises(ValueError, match='out is'):
        cirq.targeted_left_multiply(left_matrix=m, right_target=common, out=common, target_axes=[2])
    with pytest.raises(ValueError, match='out is'):
        cirq.targeted_left_multiply(left_matrix=m, right_target=common, out=m, target_axes=[2])