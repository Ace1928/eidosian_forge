from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('method', methods)
def test_three_nodes(self, method):
    pytest.importorskip('scipy')
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (2, 3, 1)], weight='spam')
    order = nx.spectral_ordering(G, weight='spam', method=method)
    assert set(order) == set(G)
    assert {1, 3} in (set(order[:-1]), set(order[1:]))