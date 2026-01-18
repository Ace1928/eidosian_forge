import pytest
import numpy as np
from cirq.testing import sample_gates
import cirq
@pytest.mark.parametrize('target_bitsize, phase_state', [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('ancilla_bitsize', [1, 4])
def test_phase_using_dirty_ancilla(target_bitsize, phase_state, ancilla_bitsize):
    g = sample_gates.PhaseUsingDirtyAncilla(phase_state, target_bitsize, ancilla_bitsize)
    q = cirq.LineQubit.range(target_bitsize)
    qubit_order = cirq.QubitOrder.explicit(q, fallback=cirq.QubitOrder.DEFAULT)
    decomposed_circuit = cirq.Circuit(cirq.decompose_once(g.on(*q)))
    decomposed_unitary = decomposed_circuit.unitary(qubit_order=qubit_order)
    phase_matrix = np.eye(2 ** target_bitsize)
    phase_matrix[phase_state, phase_state] = -1
    np.testing.assert_allclose(g.narrow_unitary(), phase_matrix)
    np.testing.assert_allclose(decomposed_unitary, np.kron(phase_matrix, np.eye(2 ** ancilla_bitsize)), atol=1e-05)