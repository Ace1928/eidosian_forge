from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_unroll_circuit_op_no_tags():
    q = cirq.LineQubit.range(2)
    op_list = [cirq.X(q[0]), cirq.Y(q[1])]
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(op_list))
    op2 = op1.with_tags('custom tag')
    op3 = op1.with_tags(MAPPED_CIRCUIT_OP_TAG)
    c = cirq.Circuit(op1, op2, op3)
    for unroller in [cirq.unroll_circuit_op, cirq.unroll_circuit_op_greedy_earliest, cirq.unroll_circuit_op_greedy_frontier]:
        cirq.testing.assert_same_circuits(unroller(c, tags_to_check=None), cirq.Circuit([op_list] * 3))
        cirq.testing.assert_same_circuits(unroller(c), cirq.Circuit([op1, op2, op_list]))
        cirq.testing.assert_same_circuits(unroller(c, tags_to_check=('custom tag',)), cirq.Circuit([op1, op_list, op3]))
        cirq.testing.assert_same_circuits(unroller(c, tags_to_check=('custom tag', MAPPED_CIRCUIT_OP_TAG)), cirq.Circuit([op1, op_list, op_list]))