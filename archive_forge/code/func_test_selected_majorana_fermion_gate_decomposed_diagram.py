import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_selected_majorana_fermion_gate_decomposed_diagram():
    selection_bitsize, target_bitsize = (2, 3)
    gate = cirq_ft.SelectedMajoranaFermionGate(cirq_ft.SelectionRegister('selection', selection_bitsize, target_bitsize), target_gate=cirq.X)
    greedy_mm = cirq.GreedyQubitManager(prefix='_a', maximize_reuse=True)
    g = cirq_ft.testing.GateHelper(gate)
    context = cirq.DecompositionContext(greedy_mm)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation, context=context))
    ancillas = sorted(set(circuit.all_qubits()) - set(g.operation.qubits))
    qubits = np.concatenate([g.quregs['control'], [q for qs in zip(g.quregs['selection'], ancillas[1:]) for q in qs], ancillas[0:1], g.quregs['target']])
    cirq.testing.assert_has_diagram(circuit, '\ncontrol: ──────@───@──────────────────────────────────────@───────────@──────\n               │   │                                      │           │\nselection0: ───┼───(0)────────────────────────────────────┼───────────@──────\n               │   │                                      │           │\n_a_1: ─────────┼───And───@─────────────@───────────@──────X───@───@───And†───\n               │         │             │           │          │   │\nselection1: ───┼─────────(0)───────────┼───────────@──────────┼───┼──────────\n               │         │             │           │          │   │\n_a_2: ─────────┼─────────And───@───@───X───@───@───And†───────┼───┼──────────\n               │               │   │       │   │              │   │\n_a_0: ─────────X───────────────X───┼───@───X───┼───@──────────X───┼───@──────\n                                   │   │       │   │              │   │\ntarget0: ──────────────────────────X───@───────┼───┼──────────────┼───┼──────\n                                               │   │              │   │\ntarget1: ──────────────────────────────────────X───@──────────────┼───┼──────\n                                                                  │   │\ntarget2: ─────────────────────────────────────────────────────────X───@──────    ', qubit_order=qubits)