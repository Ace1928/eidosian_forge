import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_empty_density_matrix():
    matrix = np.zeros(shape=())
    np.testing.assert_almost_equal(cirq.sample_density_matrix(matrix, []), [[]])