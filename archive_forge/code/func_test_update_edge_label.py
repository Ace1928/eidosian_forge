import random
import pytest
import cirq.contrib.graph_device as ccgd
def test_update_edge_label():
    edge = frozenset(range(3))
    graph = ccgd.UndirectedHypergraph(labelled_edges={edge: 'a'})
    assert graph.labelled_edges[edge] == 'a'
    graph.add_edge(edge, 'b')
    assert graph.labelled_edges[edge] == 'b'