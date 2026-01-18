import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_no_indices():
    matrix = cirq.to_valid_density_matrix(0, 3)
    bits = cirq.sample_density_matrix(matrix, [])
    np.testing.assert_almost_equal(bits, np.zeros(shape=(1, 0)))