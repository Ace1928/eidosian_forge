import networkx as nx
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def _get_circuit(width, height, rs, p=2):
    graph = nx.grid_2d_graph(width, height)
    nx.set_edge_attributes(graph, name='weight', values={e: np.round(rs.uniform(), 2) for e in graph.edges})
    qubits = [cirq.GridQubit(*n) for n in graph]
    circuit = cirq.Circuit(cirq.H.on_each(qubits), [(ccq.get_grid_moments(graph), cirq.Moment([cirq.rx(0.456).on_each(qubits)])) for _ in range(p)])
    return (circuit, qubits)