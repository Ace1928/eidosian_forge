import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_step_result_bloch_vector():
    q0, q1 = cirq.LineQubit.range(2)
    step_result = BasicStateVector({q0: 0, q1: 1})
    bloch1 = np.array([0, 0, -1])
    bloch0 = np.array([0, 0, 1])
    np.testing.assert_array_almost_equal(bloch1, step_result.bloch_vector_of(q1))
    np.testing.assert_array_almost_equal(bloch0, step_result.bloch_vector_of(q0))