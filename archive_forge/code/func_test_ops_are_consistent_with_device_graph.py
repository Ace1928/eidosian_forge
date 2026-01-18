import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_ops_are_consistent_with_device_graph():
    device_graph = ccr.get_linear_device_graph(3)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.ZZ(qubits[0], qubits[2]))
    assert not ccr.ops_are_consistent_with_device_graph(circuit.all_operations(), device_graph)
    assert not ccr.ops_are_consistent_with_device_graph([cirq.X(cirq.GridQubit(0, 0))], device_graph)