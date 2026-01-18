import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite.cluster import cc_dot, cc_max, cc_min
def test_ra_clustering_square():
    G = nx.path_graph(4)
    G.add_edge(0, 3)
    assert bipartite.robins_alexander_clustering(G) == 1.0