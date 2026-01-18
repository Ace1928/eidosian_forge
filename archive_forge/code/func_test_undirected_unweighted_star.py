import pytest
import networkx as nx
def test_undirected_unweighted_star(self):
    G = nx.star_graph(2)
    grc = nx.local_reaching_centrality
    assert grc(G, 1, weight=None, normalized=False) == 0.75