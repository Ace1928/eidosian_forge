import itertools
import numpy as np
import pytest
import scipy.linalg
import cirq
@pytest.mark.parametrize('coefficients,exponent', itertools.product(((0, 0, 0, 0), (-1, 0, 0, 0), (0.5, 0, 0, 0), (0.5j, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (0, -1, 0, 0), (0, 0.5, 0, 0), (0, 0.5j, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 0, -1, 0), (0, 0, 0.5, 0), (0, 0, 0.5j, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 0, -1), (0, 0, 0, 0.5), (0, 0, 0, 0.5j), (0, 0, 0, 1), (0, 0, 0, 2), (0, -1, 0, -1), (0, 1, 0, 1j), (0, 0.5, 0, 0.5), (0, 0.5j, 0, 0.5j), (0, 0.5, 0, 0.5j), (0, 1, 0, 1), (0, 2, 0, 2), (0, 0.5, 0.5, 0.5), (0, 1, 1, 1), (0, 1.1j, 0.5 - 0.4j, 0.9), (0.7j, 1.1j, 0.5 - 0.4j, 0.9), (0.25, 0.25, 0.25, 0.25), (0.25j, 0.25j, 0.25j, 0.25j), (0.4, 0, 0.5, 0), (1, 2, 3, 4), (-1, -2, -3, -4), (-1, -2, 3, 4), (1j, 2j, 3j, 4j), (1j, 2j, 3, 4)), (0, 1, 2, 3, 4, 5, 100, 101)))
def test_pow_pauli_combination(coefficients, exponent):
    i = cirq.PAULI_BASIS['I']
    x = cirq.PAULI_BASIS['X']
    y = cirq.PAULI_BASIS['Y']
    z = cirq.PAULI_BASIS['Z']
    ai, ax, ay, az = coefficients
    matrix = ai * i + ax * x + ay * y + az * z
    expected_result = np.linalg.matrix_power(matrix, exponent)
    bi, bx, by, bz = cirq.pow_pauli_combination(ai, ax, ay, az, exponent)
    result = bi * i + bx * x + by * y + bz * z
    assert np.allclose(result, expected_result)