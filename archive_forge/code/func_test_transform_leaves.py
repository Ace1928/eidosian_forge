from typing import cast
import pytest
import cirq
def test_transform_leaves():
    gs = [cirq.testing.SingleQubitGate() for _ in range(10)]
    operations = [cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i))]) for i in range(10)]
    expected = [cirq.GateOperation(gs[i], [cirq.NamedQubit(str(i) + 'a')]) for i in range(10)]

    def move_left(op: cirq.GateOperation):
        return cirq.GateOperation(op.gate, [cirq.NamedQubit(cast(cirq.NamedQubit, q).name + 'a') for q in op.qubits])

    def move_tree_left_freeze(root):
        return cirq.freeze_op_tree(cirq.transform_op_tree(root, move_left))
    assert move_tree_left_freeze([[[]]]) == (((),),)
    assert move_tree_left_freeze(operations[0]) == expected[0]
    assert move_tree_left_freeze(operations) == tuple(expected)
    assert move_tree_left_freeze((operations[0], operations[1:5], operations[5:])) == (expected[0], tuple(expected[1:5]), tuple(expected[5:]))