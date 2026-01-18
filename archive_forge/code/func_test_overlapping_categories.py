import pytest
import cirq
def test_overlapping_categories():
    a, b, c, d = cirq.LineQubit.range(4)
    result = cirq.stratified_circuit(cirq.Circuit(cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]), cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.CNOT(c, d)]), cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)])), categories=[lambda op: len(op.qubits) == 1 and (not isinstance(op.gate, cirq.XPowGate)), lambda op: len(op.qubits) == 1 and (not isinstance(op.gate, cirq.ZPowGate))])
    cirq.testing.assert_same_circuits(result, cirq.Circuit(cirq.Moment([cirq.Y(b), cirq.Z(c)]), cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.CNOT(a, b), cirq.CNOT(c, d)]), cirq.Moment([cirq.Y(b), cirq.Z(c)]), cirq.Moment([cirq.X(a)])))