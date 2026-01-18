import copy
import pytest
import numpy as np
import cirq
@pytest.mark.parametrize('state', STATES_TO_PREPARE)
@pytest.mark.parametrize('use_iswap_inv', [True, False])
def test_prepare_two_qubit_state_using_iswap(state, use_iswap_inv):
    state = cirq.to_valid_state_vector(state, num_qubits=2)
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.prepare_two_qubit_state_using_iswap(*q, state, use_iswap_inv=use_iswap_inv))
    iswap_gate = cirq.ISWAP_INV if use_iswap_inv else cirq.ISWAP
    ops_iswap = [*circuit.findall_operations(lambda op: op.gate == iswap_gate)]
    ops_2q = [*circuit.findall_operations(lambda op: cirq.num_qubits(op) > 1)]
    assert ops_iswap == ops_2q
    assert len(ops_iswap) <= 1
    assert cirq.allclose_up_to_global_phase(circuit.final_state_vector(ignore_terminal_measurements=False, dtype=np.complex64), state)