import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_phased_x_z_tolerance_xy():
    c, s = (np.cos(0.01), np.sin(0.01))
    xy = np.array([[c, -s], [s, c]])
    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(xy, atol=0.1)
    assert len(optimized_away) == 0
    kept = cirq.single_qubit_matrix_to_phased_x_z(xy, atol=0.0001)
    assert len(kept) == 1