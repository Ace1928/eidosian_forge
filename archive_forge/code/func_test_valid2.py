import pytest
import networkx
def test_valid2(self):
    G = networkx.random_clustered_graph([(1, 2), (2, 1), (1, 1), (1, 1), (1, 1), (2, 0)])
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 10