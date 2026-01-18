import networkx as nx
import pytest
import cirq
def test_valid_circuit():
    circuit = construct_valid_circuit()
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(circuit)
    mapped_circuit = circuit.transform_qubits(mapping)
    device.validate_circuit(mapped_circuit)