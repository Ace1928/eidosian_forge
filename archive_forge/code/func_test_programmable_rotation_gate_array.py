from typing import Tuple
from numpy.typing import NDArray
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('angles', [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[3, 4, 5], [10, 11, 12]]])
@pytest.mark.parametrize('kappa', [*range(1, 12)])
@pytest.mark.parametrize('constructor', [construct_custom_prga, construct_prga_with_phase, construct_prga_with_identity])
@allow_deprecated_cirq_ft_use_in_tests
def test_programmable_rotation_gate_array(angles, kappa, constructor):
    rotation_gate = cirq.X
    programmable_rotation_gate = constructor(*angles, kappa=kappa, rotation_gate=rotation_gate)
    greedy_mm = cirq.GreedyQubitManager(prefix='_a')
    g = cirq_ft.testing.GateHelper(programmable_rotation_gate, context=cirq.DecompositionContext(greedy_mm))
    decomposed_circuit = cirq.Circuit(cirq.I.on_each(*g.all_qubits)) + g.decomposed_circuit
    interleaved_unitaries = [programmable_rotation_gate.interleaved_unitary(i, **g.quregs) for i in range(len(angles) - 1)]
    rotations_and_unitary_registers = cirq_ft.Signature([*programmable_rotation_gate.rotations_target, *programmable_rotation_gate.interleaved_unitary_target])
    rotations_and_unitary_qubits = infra.merge_qubits(rotations_and_unitary_registers, **g.quregs)
    simulator = cirq.Simulator(dtype=np.complex128)

    def rotation_ops(theta: int) -> cirq.OP_TREE:
        for i, b in enumerate(bin(theta)[2:][::-1]):
            if b == '1':
                yield cirq.pow(rotation_gate.on(*g.quregs['rotations_target']), 1 / 2 ** (1 + i))
    for selection_integer in range(len(angles[0])):
        qubit_vals = {x: 0 for x in g.all_qubits}
        qubit_vals.update(zip(g.quregs['selection'], iter_bits(selection_integer, g.r.get_left('selection').total_bits())))
        initial_state = [qubit_vals[x] for x in g.all_qubits]
        result = simulator.simulate(decomposed_circuit, initial_state=initial_state, qubit_order=g.all_qubits)
        ru_state_vector = cirq.sub_state_vector(result.final_state_vector, keep_indices=[g.all_qubits.index(q) for q in rotations_and_unitary_qubits])
        expected_circuit = cirq.Circuit([[rotation_ops(angles[i][selection_integer]), u] for i, u in enumerate(interleaved_unitaries)], rotation_ops(angles[-1][selection_integer]))
        expected_ru_state_vector = simulator.simulate(expected_circuit, qubit_order=rotations_and_unitary_qubits).final_state_vector
        cirq.testing.assert_allclose_up_to_global_phase(ru_state_vector, expected_ru_state_vector, atol=1e-08)
        ancilla_indices = [g.all_qubits.index(q) for q in g.all_qubits if q not in rotations_and_unitary_qubits]
        ancilla_state_vector = cirq.sub_state_vector(result.final_state_vector, keep_indices=ancilla_indices)
        expected_ancilla_state_vector = cirq.quantum_state([initial_state[x] for x in ancilla_indices], qid_shape=(2,) * len(ancilla_indices), dtype=np.complex128).state_vector()
        cirq.testing.assert_allclose_up_to_global_phase(ancilla_state_vector, expected_ancilla_state_vector, atol=1e-08)