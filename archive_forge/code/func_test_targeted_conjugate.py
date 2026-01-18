import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_conjugate():
    a = np.reshape([0, 1, 2j, 3j], (2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0])
    expected = np.einsum('ij,jklm,ln->iknm', a, b, np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)
    result = cirq.targeted_conjugate_about(a, b, [1])
    expected = np.einsum('ij,kjlm,mn->kiln', a, b, np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)