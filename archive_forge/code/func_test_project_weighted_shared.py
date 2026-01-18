import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_project_weighted_shared(self):
    edges = [('A', 'B', 2), ('A', 'C', 1), ('B', 'C', 1), ('B', 'D', 1), ('B', 'E', 2), ('E', 'F', 1)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.weighted_projected_graph(self.G, 'ABCDEF')
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']
    edges = [('A', 'B', 3), ('A', 'E', 1), ('A', 'C', 1), ('A', 'D', 1), ('B', 'E', 1), ('B', 'C', 1), ('B', 'D', 1), ('C', 'D', 1)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.weighted_projected_graph(self.N, 'ABCDE')
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']