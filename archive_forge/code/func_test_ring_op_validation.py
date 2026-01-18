import pytest
import networkx as nx
import cirq
def test_ring_op_validation():
    directed_device = cirq.testing.construct_ring_device(5, directed=True)
    undirected_device = cirq.testing.construct_ring_device(5, directed=False)
    with pytest.raises(ValueError, match='Qubits not on device'):
        directed_device.validate_operation(cirq.X(cirq.LineQubit(5)))
    with pytest.raises(ValueError, match='Qubits not on device'):
        undirected_device.validate_operation(cirq.X(cirq.LineQubit(5)))
    with pytest.raises(ValueError, match='Qubit pair is not a valid edge on device'):
        undirected_device.validate_operation(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)))
    with pytest.raises(ValueError, match='Qubit pair is not a valid edge on device'):
        directed_device.validate_operation(cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)))
    undirected_device.validate_operation(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))
    undirected_device.validate_operation(cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)))
    directed_device.validate_operation(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))