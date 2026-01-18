from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
@pytest.mark.parametrize('shift_left', [True, False])
def test_two_qubit_multiplexor_to_circuit(shift_left):
    a, b, c = cirq.LineQubit.range(3)
    u1 = cirq.testing.random_unitary(4)
    u2 = cirq.testing.random_unitary(4)
    d_ud, ud_ops = _two_qubit_multiplexor_to_ops(a, b, c, u1, u2, shift_left=shift_left)
    expected = block_diag(u1, u2)
    diagonal = np.kron(np.eye(2), d_ud) if d_ud is not None else np.eye(8)
    actual = cirq.Circuit(ud_ops).unitary(qubits_that_should_be_present=[a, b, c]) @ diagonal
    cirq.testing.assert_allclose_up_to_global_phase(expected, actual, atol=1e-08)