import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_seed():
    density_matrix = 0.5 * np.eye(2)
    samples = cirq.sample_density_matrix(density_matrix, [0], repetitions=10, seed=1234)
    assert np.array_equal(samples, [[False], [True], [False], [True], [True], [False], [False], [True], [True], [True]])
    samples = cirq.sample_density_matrix(density_matrix, [0], repetitions=10, seed=np.random.RandomState(1234))
    assert np.array_equal(samples, [[False], [True], [False], [True], [True], [False], [False], [True], [True], [True]])