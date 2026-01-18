import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_partial_trace_non_kron():
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 0] = 1
    tensor[1, 1, 1, 1] = 4
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0]), np.array([[1, 0], [0, 4]]))