import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_negative_repetitions():
    matrix = cirq.to_valid_density_matrix(0, 3)
    with pytest.raises(ValueError, match='-1'):
        cirq.sample_density_matrix(matrix, [1], repetitions=-1)