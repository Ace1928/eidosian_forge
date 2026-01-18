import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('initial_state, is_valid', list(zip(VALID_INITIAL_STATES + INVALID_INITIAL_STATES, [True] * len(VALID_INITIAL_STATES) + [False] * len(INVALID_INITIAL_STATES))))
@pytest.mark.parametrize('unitary_matrix', [cirq.testing.random_unitary(4) for _ in range(5)])
@pytest.mark.parametrize('allow_partial_czs', [True, False])
def test_two_qubit_matrix_to_cz_isometry(initial_state, is_valid, unitary_matrix, allow_partial_czs):
    a, b, c = cirq.LineQubit.range(3)
    decomposed_ops = cirq.two_qubit_matrix_to_cz_isometry(a, b, unitary_matrix, allow_partial_czs=allow_partial_czs)
    circuit = cirq.Circuit(decomposed_ops)
    ops_cz = [*circuit.findall_operations(lambda op: isinstance(op.gate, cirq.CZPowGate))]
    ops_2q = [*circuit.findall_operations(lambda op: cirq.num_qubits(op) > 1)]
    assert ops_cz == ops_2q
    assert len(ops_cz) <= 2
    sim = cirq.Simulator()
    base_circuit = cirq.Circuit(cirq.StatePreparationChannel(initial_state).on(a, b), cirq.CNOT(b, c))
    original_circuit = base_circuit + cirq.Circuit(cirq.MatrixGate(unitary_matrix).on(a, b))
    decomposed_circuit = base_circuit + circuit
    original_final_state_vector = sim.simulate(original_circuit).final_state_vector
    decomposed_final_state_vector = sim.simulate(decomposed_circuit).final_state_vector
    assert cirq.allclose_up_to_global_phase(original_final_state_vector, decomposed_final_state_vector, atol=1e-06) is is_valid