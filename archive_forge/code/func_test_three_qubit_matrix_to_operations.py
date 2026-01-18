from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
@_skip_if_scipy(version_is_greater_than_1_5_0=False)
@pytest.mark.parametrize('u', [cirq.testing.random_unitary(8), np.eye(8), cirq.ControlledGate(cirq.ISWAP)._unitary_(), cirq.CCX._unitary_()])
def test_three_qubit_matrix_to_operations(u):
    a, b, c = cirq.LineQubit.range(3)
    operations = cirq.three_qubit_matrix_to_operations(a, b, c, u)
    final_circuit = cirq.Circuit(operations)
    final_unitary = final_circuit.unitary(qubits_that_should_be_present=[a, b, c])
    cirq.testing.assert_allclose_up_to_global_phase(u, final_unitary, atol=1e-09)
    num_two_qubit_gates = len([op for op in list(final_circuit.all_operations()) if isinstance(op.gate, (cirq.CZPowGate, cirq.CNotPowGate))])
    assert num_two_qubit_gates <= 20, f'expected at most 20 CZ/CNOTs got {num_two_qubit_gates}'