from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_unroll_circuit_op_and_variants():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(q[0]), cirq.CNOT(q[0], q[1]), cirq.X(q[0]))
    cirq.testing.assert_has_diagram(c, '\n0: ───X───@───X───\n          │\n1: ───────X───────\n')
    mapped_circuit = cirq.map_operations(c, lambda op, i: [cirq.Z(q[1])] * 2 if op.gate == cirq.CNOT else op)
    mapped_circuit_deep = cirq.Circuit([cirq.Moment(cirq.CircuitOperation(cirq.FrozenCircuit(m))) for m in mapped_circuit[:-1]], mapped_circuit[-1])
    cirq.testing.assert_has_diagram(mapped_circuit_deep, "\n0: ───[ 0: ───X─── ]────────────────────────────────────────────────────────────X───\n\n1: ────────────────────[ 1: ───[ 1: ───Z───Z─── ]['<mapped_circuit_op>']─── ]───────\n")
    for unroller in [cirq.unroll_circuit_op_greedy_earliest, cirq.unroll_circuit_op_greedy_frontier, cirq.unroll_circuit_op]:
        cirq.testing.assert_same_circuits(unroller(mapped_circuit), unroller(mapped_circuit_deep, deep=True, tags_to_check=None))
        cirq.testing.assert_has_diagram(unroller(mapped_circuit_deep, deep=True), '\n0: ───[ 0: ───X─── ]────────────────────────X───\n\n1: ────────────────────[ 1: ───Z───Z─── ]───────\n            ')
    cirq.testing.assert_has_diagram(cirq.unroll_circuit_op(mapped_circuit), '\n0: ───X───────────X───\n\n1: ───────Z───Z───────\n')
    cirq.testing.assert_has_diagram(cirq.unroll_circuit_op_greedy_earliest(mapped_circuit), '\n0: ───X───────X───\n\n1: ───Z───Z───────\n')
    cirq.testing.assert_has_diagram(cirq.unroll_circuit_op_greedy_frontier(mapped_circuit), '\n0: ───X───────X───\n\n1: ───────Z───Z───\n')