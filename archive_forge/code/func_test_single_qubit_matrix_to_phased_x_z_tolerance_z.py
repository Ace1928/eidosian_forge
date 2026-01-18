import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
def test_single_qubit_matrix_to_phased_x_z_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])
    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(z, atol=0.1)
    assert len(optimized_away) == 0
    kept = cirq.single_qubit_matrix_to_phased_x_z(z, atol=0.0001)
    assert len(kept) == 1