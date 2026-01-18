import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_phxz_tolerance_xy():
    c, s = (np.cos(0.01), np.sin(0.01))
    xy = np.array([[c, -s], [s, c]])
    optimized_away = cirq.single_qubit_matrix_to_phxz(xy, atol=0.1)
    assert optimized_away is None
    kept = cirq.single_qubit_matrix_to_phxz(xy, atol=0.0001)
    assert kept is not None