import pytest
import networkx as nx
import cirq
def test_ring_device():
    undirected_device = cirq.testing.construct_ring_device(5)
    undirected_device_graph = undirected_device.metadata.nx_graph
    assert all((q in undirected_device_graph.nodes for q in cirq.LineQubit.range(5)))
    isomorphism_class = nx.Graph()
    edges = [(cirq.LineQubit(i % 5), cirq.LineQubit((i + 1) % 5)) for i in range(5)]
    isomorphism_class.add_edges_from(edges)
    assert nx.is_isomorphic(isomorphism_class, undirected_device_graph)
    directed_device = cirq.testing.construct_ring_device(5, directed=True)
    directed_device_graph = directed_device.metadata.nx_graph
    assert all((q in directed_device_graph.nodes for q in cirq.LineQubit.range(5)))
    isomorphism_class = nx.DiGraph()
    edges = [(cirq.LineQubit(i % 5), cirq.LineQubit((i + 1) % 5)) for i in range(5)]
    isomorphism_class.add_edges_from(edges)
    assert nx.is_isomorphic(isomorphism_class, directed_device_graph)