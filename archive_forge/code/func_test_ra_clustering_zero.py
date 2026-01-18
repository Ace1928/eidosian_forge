import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.cluster import cc_dot, cc_max, cc_min
def test_ra_clustering_zero():
    G = nx.Graph()
    assert bipartite.robins_alexander_clustering(G) == 0
    G.add_nodes_from(range(4))
    assert bipartite.robins_alexander_clustering(G) == 0
    G.add_edges_from([(0, 1), (2, 3), (3, 4)])
    assert bipartite.robins_alexander_clustering(G) == 0
    G.add_edge(1, 2)
    assert bipartite.robins_alexander_clustering(G) == 0