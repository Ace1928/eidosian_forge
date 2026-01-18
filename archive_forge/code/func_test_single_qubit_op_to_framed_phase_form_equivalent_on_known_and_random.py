import random
from typing import Sequence
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('mat', [np.eye(2), cirq.unitary(cirq.H), cirq.unitary(cirq.X), cirq.unitary(cirq.X ** 0.5), cirq.unitary(cirq.Y), cirq.unitary(cirq.Z), cirq.unitary(cirq.Z ** 0.5), _random_unitary_with_close_eigenvalues()] + [cirq.testing.random_unitary(2) for _ in range(10)])
def test_single_qubit_op_to_framed_phase_form_equivalent_on_known_and_random(mat):
    u, t, g = cirq.single_qubit_op_to_framed_phase_form(mat)
    z = np.diag([g, g * t])
    assert np.allclose(mat, np.conj(u.T).dot(z).dot(u))