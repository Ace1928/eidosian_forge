import copy
import pytest
import numpy as np
import cirq
@pytest.mark.parametrize('state', STATES_TO_PREPARE)
def test_prepare_two_qubit_state_using_cz(state):
    state = cirq.to_valid_state_vector(state, num_qubits=2)
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.prepare_two_qubit_state_using_cz(*q, state))
    ops_cz = [*circuit.findall_operations(lambda op: op.gate == cirq.CZ)]
    ops_2q = [*circuit.findall_operations(lambda op: cirq.num_qubits(op) > 1)]
    assert ops_cz == ops_2q
    assert len(ops_cz) <= 1
    assert cirq.allclose_up_to_global_phase(circuit.final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64), state)