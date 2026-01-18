import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('left_part_lens,right_part_lens', set((key[1:] for key in circuit_diagrams)))
def test_shift_swap_network_gate_diagrams(left_part_lens, right_part_lens):
    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(gate(*qubits))
    diagram = circuit_diagrams['undecomposed', left_part_lens, right_part_lens]
    cirq.testing.assert_has_diagram(circuit, diagram)
    cca.expose_acquaintance_gates(circuit)
    diagram = circuit_diagrams['decomposed', left_part_lens, right_part_lens]
    cirq.testing.assert_has_diagram(circuit, diagram)