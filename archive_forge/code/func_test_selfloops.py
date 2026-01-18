import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('graph_type', [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_selfloops(graph_type):
    G = nx.complete_graph(3, create_using=graph_type)
    G.add_edge(0, 0)
    assert nodes_equal(nx.nodes_with_selfloops(G), [0])
    assert edges_equal(nx.selfloop_edges(G), [(0, 0)])
    assert edges_equal(nx.selfloop_edges(G, data=True), [(0, 0, {})])
    assert nx.number_of_selfloops(G) == 1