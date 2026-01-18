import pytest
import networkx as nx
def test_triangle_and_edge(self):
    G = nx.cycle_graph(3)
    G.add_edge(0, 4, weight=2)
    assert nx.clustering(G)[0] == 1 / 3
    np.testing.assert_allclose(nx.clustering(G, weight='weight')[0], 1 / 6)
    np.testing.assert_allclose(nx.clustering(G, 0, weight='weight'), 1 / 6)