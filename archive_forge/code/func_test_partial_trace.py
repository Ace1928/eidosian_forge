import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace():
    a = np.reshape(np.arange(4), (2, 2))
    b = np.reshape(np.arange(9) + 4, (3, 3))
    c = np.reshape(np.arange(16) + 13, (4, 4))
    tr_a = np.trace(a)
    tr_b = np.trace(b)
    tr_c = np.trace(c)
    tensor = np.reshape(np.kron(a, np.kron(b, c)), (2, 3, 4, 2, 3, 4))
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, []), tr_a * tr_b * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0]), a * tr_b * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [1]), b * tr_a * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [2]), c * tr_a * tr_b)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0, 1]), np.reshape(np.kron(a, b), (2, 3, 2, 3)) * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [1, 2]), np.reshape(np.kron(b, c), (3, 4, 3, 4)) * tr_a)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0, 2]), np.reshape(np.kron(a, c), (2, 4, 2, 4)) * tr_b)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0, 1, 2]), tensor)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [1, 0]), np.reshape(np.kron(b, a), (3, 2, 3, 2)) * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [2, 0, 1]), np.reshape(np.kron(c, np.kron(a, b)), (4, 2, 3, 4, 2, 3)))