import pytest
import networkx as nx
def test_undirected_weighted_star(self):
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2)])
    centrality = nx.local_reaching_centrality(G, 1, normalized=False, weight='weight')
    assert centrality == 1.5