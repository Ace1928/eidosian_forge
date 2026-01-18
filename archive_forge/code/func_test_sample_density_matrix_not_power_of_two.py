import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_not_power_of_two():
    with pytest.raises(ValueError, match='power of two'):
        cirq.sample_density_matrix(np.ones((3, 3)) / 3, [1])
    with pytest.raises(ValueError, match='power of two'):
        cirq.sample_density_matrix(np.ones((2, 3, 2, 3)) / 6, [1])