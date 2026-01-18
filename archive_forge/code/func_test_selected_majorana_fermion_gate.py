import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('selection_bitsize, target_bitsize', [(2, 4), pytest.param(3, 8, marks=pytest.mark.slow), pytest.param(4, 9, marks=pytest.mark.slow)])
@pytest.mark.parametrize('target_gate', [cirq.X, cirq.Y])
@allow_deprecated_cirq_ft_use_in_tests
def test_selected_majorana_fermion_gate(selection_bitsize, target_bitsize, target_gate):
    gate = cirq_ft.SelectedMajoranaFermionGate(cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), target_gate=target_gate)
    g = cirq_ft.testing.GateHelper(gate)
    assert len(g.all_qubits) <= infra.total_bits(gate.signature) + selection_bitsize + 1
    sim = cirq.Simulator(dtype=np.complex128)
    for n in range(target_bitsize):
        qubit_vals = {q: 0 for q in g.operation.qubits}
        qubit_vals.update({c: 1 for c in g.quregs['control']})
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(n, selection_bitsize)))
        initial_state = [qubit_vals[x] for x in g.operation.qubits]
        result = sim.simulate(g.circuit, initial_state=initial_state, qubit_order=g.operation.qubits)
        final_target_state = cirq.sub_state_vector(result.final_state_vector, keep_indices=[g.operation.qubits.index(q) for q in g.quregs['target']])
        expected_target_state = cirq.Circuit([cirq.Z(q) for q in g.quregs['target'][:n]], target_gate(g.quregs['target'][n]), [cirq.I(q) for q in g.quregs['target'][n + 1:]]).final_state_vector(qubit_order=g.quregs['target'])
        cirq.testing.assert_allclose_up_to_global_phase(expected_target_state, final_target_state, atol=1e-06)