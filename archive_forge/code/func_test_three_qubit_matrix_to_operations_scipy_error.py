from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
@_skip_if_scipy(version_is_greater_than_1_5_0=True)
def test_three_qubit_matrix_to_operations_scipy_error():
    a, b, c = cirq.LineQubit.range(3)
    with pytest.raises(ImportError, match='three_qubit.*1.5.0+'):
        cirq.three_qubit_matrix_to_operations(a, b, c, np.eye(8))