from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_merge_and_replace_inefficient_component():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.CNOT(*q)), cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.CZ(*q).with_tags('no_compile')), cirq.Moment(cirq.Z.on_each(*q)), cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.CNOT(*q)), cirq.Moment(cirq.CNOT(*q)), cirq.Moment(cirq.Z.on_each(*q)), cirq.Moment(cirq.X(q[0])), cirq.Moment(cirq.CNOT(*q)), cirq.measure(q[0], key='m'), cirq.X(q[1]).with_classical_controls('m'))
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───X───@───X───@['no_compile']───Z───X───@───@───Z───X───@───M───────\n          │       │                         │   │           │   ║\n1: ───────X───────@─────────────────Z───────X───X───Z───────X───╫───X───\n                                                                ║   ║\nm: ═════════════════════════════════════════════════════════════@═══^═══\n")
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=ExampleCXTargetGateset(), context=cirq.TransformerContext(tags_to_ignore=('no_compile',)))
    cirq.testing.assert_has_diagram(c_new, "\n0: ───X───@───X───@['no_compile']───X───@───Y───@───Z───M───────\n          │       │                     │       │       ║\n1: ───────X───────@─────────────────X───X───Y───X───Z───╫───X───\n                                                        ║   ║\nm: ═════════════════════════════════════════════════════@═══^═══\n")