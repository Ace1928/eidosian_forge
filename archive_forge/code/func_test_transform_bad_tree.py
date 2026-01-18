from typing import cast
import pytest
import cirq
def test_transform_bad_tree():
    with pytest.raises(TypeError):
        _ = list(cirq.transform_op_tree(None))
    with pytest.raises(TypeError):
        _ = list(cirq.transform_op_tree(5))
    with pytest.raises(TypeError):
        _ = list(cirq.flatten_op_tree(cirq.transform_op_tree([cirq.GateOperation(cirq.Gate(), [cirq.NamedQubit('q')]), (4,)])))