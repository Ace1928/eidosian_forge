import numpy as np
import pytest
import cirq
def test_von_neumann_entropy():
    assert cirq.von_neumann_entropy(np.array([[1]])) == 0
    assert cirq.von_neumann_entropy(0.5 * np.array([1, 0, 0, 1] * np.array([[1], [0], [0], [1]]))) == 0
    assert cirq.von_neumann_entropy(np.array([[0.5, 0], [0, 0.5]])) == 1
    res = cirq.testing.random_unitary(2)
    first_column = res[:, 0]
    first_density_matrix = 0.1 * np.outer(first_column, np.conj(first_column))
    second_column = res[:, 1]
    second_density_matrix = 0.9 * np.outer(second_column, np.conj(second_column))
    assert np.isclose(cirq.von_neumann_entropy(first_density_matrix + second_density_matrix), 0.4689, atol=0.0001)
    assert np.isclose(cirq.von_neumann_entropy(np.diag([0, 0, 0.1, 0, 0.2, 0.3, 0.4, 0])), 1.8464, atol=0.0001)
    probs = np.random.exponential(size=N)
    probs /= np.sum(probs)
    mat = U @ (probs * U).T.conj()
    np.testing.assert_allclose(cirq.von_neumann_entropy(mat), -np.sum(probs * np.log(probs) / np.log(2)))
    assert cirq.von_neumann_entropy(cirq.quantum_state(np.array([[0.5, 0], [0, 0.5]]), qid_shape=(2,))) == 1
    assert cirq.von_neumann_entropy(cirq.quantum_state(np.array([[0.5, 0.5], [0.5, 0.5]]), qid_shape=(2, 2))) == 0