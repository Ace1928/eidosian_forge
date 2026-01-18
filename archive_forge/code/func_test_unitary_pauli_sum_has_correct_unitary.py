import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
@pytest.mark.parametrize('psum, expected_unitary', ((np.sqrt(0.5) * (cirq.X(q0) + cirq.Z(q0)), np.sqrt(0.5) * np.array([[1, 1], [1, -1]])), (np.sqrt(0.5) * (cirq.X(q0) * cirq.X(q1) + cirq.Z(q1)), np.sqrt(0.5) * np.array([[1, 0, 0, 1], [0, -1, 1, 0], [0, 1, 1, 0], [1, 0, 0, -1]]))))
def test_unitary_pauli_sum_has_correct_unitary(psum, expected_unitary):
    assert cirq.has_unitary(psum)
    assert np.allclose(cirq.unitary(psum), expected_unitary)