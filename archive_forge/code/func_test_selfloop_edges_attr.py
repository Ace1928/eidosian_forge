import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('graph_type', [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_selfloop_edges_attr(graph_type):
    G = nx.complete_graph(3, create_using=graph_type)
    G.add_edge(0, 0)
    G.add_edge(1, 1, weight=2)
    assert edges_equal(nx.selfloop_edges(G, data=True), [(0, 0, {}), (1, 1, {'weight': 2})])
    assert edges_equal(nx.selfloop_edges(G, data='weight'), [(0, 0, None), (1, 1, 2)])