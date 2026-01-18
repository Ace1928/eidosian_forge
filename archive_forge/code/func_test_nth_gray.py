import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
@pytest.mark.parametrize('n, gray', [(0, 0), (1, 1), (2, 3), (3, 2), (4, 6), (5, 7), (6, 5), (7, 4), (8, 12), (9, 13), (10, 15), (11, 14), (12, 10), (13, 11), (14, 9), (15, 8)])
def test_nth_gray(n, gray):
    assert _nth_gray(n) == gray