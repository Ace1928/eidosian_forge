import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_higher_powers_of_two():
    with pytest.raises(ValueError, match='powers of two'):
        cirq.sample_density_matrix(np.ones((2, 4, 2, 4)) / 8, [1])