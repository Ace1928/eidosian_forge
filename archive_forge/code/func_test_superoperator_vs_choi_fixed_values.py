from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('superoperator, choi', ((np.eye(4), np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])), (np.diag([1, -1j, 1j, 1]), np.array([[1, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 1]])), (np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2, np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 2), (np.diag([1, 0, 0, 1]), np.diag([1, 0, 0, 1])), (np.array([[1, 0, 0, 0.36], [0, 0.8, 0, 0], [0, 0, 0.8, 0], [0, 0, 0, 0.64]]), np.array([[1, 0, 0, 0.8], [0, 0.36, 0, 0], [0, 0, 0, 0], [0.8, 0, 0, 0.64]])), (np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2, np.eye(4) / 2)))
def test_superoperator_vs_choi_fixed_values(superoperator, choi):
    recovered_choi = cirq.superoperator_to_choi(superoperator)
    assert np.allclose(recovered_choi, choi)
    recovered_superoperator = cirq.choi_to_superoperator(choi)
    assert np.allclose(recovered_superoperator, superoperator)