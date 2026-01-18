import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_project_weighted_newman(self):
    edges = [('A', 'B', 1.5), ('A', 'C', 0.5), ('B', 'C', 0.5), ('B', 'D', 1), ('B', 'E', 2), ('E', 'F', 1)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.collaboration_weighted_projected_graph(self.G, 'ABCDEF')
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']
    edges = [('A', 'B', 11 / 6.0), ('A', 'E', 1 / 2.0), ('A', 'C', 1 / 3.0), ('A', 'D', 1 / 3.0), ('B', 'E', 1 / 2.0), ('B', 'C', 1 / 3.0), ('B', 'D', 1 / 3.0), ('C', 'D', 1 / 3.0)]
    Panswer = nx.Graph()
    Panswer.add_weighted_edges_from(edges)
    P = bipartite.collaboration_weighted_projected_graph(self.N, 'ABCDE')
    assert edges_equal(list(P.edges()), Panswer.edges())
    for u, v in list(P.edges()):
        assert P[u][v]['weight'] == Panswer[u][v]['weight']