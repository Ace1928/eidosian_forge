from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_unroll_circuit_op_deep():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c = cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q2))))))
    expected = cirq.Circuit(cirq.X.on_each(q0, q1, q2))
    cirq.testing.assert_same_circuits(cirq.unroll_circuit_op(c, tags_to_check=None, deep=True), expected)
    expected = cirq.Circuit(cirq.X.on_each(q0, q1), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q2))))
    cirq.testing.assert_same_circuits(cirq.unroll_circuit_op(c, tags_to_check=None, deep=False), expected)