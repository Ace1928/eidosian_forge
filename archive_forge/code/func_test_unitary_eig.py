import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('matrix', [X, np.eye(4), np.diag(np.exp([-1j * np.pi * 1.23, -1j * np.pi * 1.23, -1j * np.pi * 1.23])), np.diag(np.exp([-0.2312j, -0.2312j, -0.2312j, -0.2312j])) + np.random.random((4, 4)) * 1e-100, _random_unitary_with_close_eigenvalues()])
def test_unitary_eig(matrix):
    d, vecs = unitary_eig(matrix)
    np.testing.assert_allclose(matrix, vecs @ np.diag(d) @ vecs.conj().T, atol=1e-14)