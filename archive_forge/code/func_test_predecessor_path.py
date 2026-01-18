import pytest
import networkx as nx
def test_predecessor_path(self):
    G = nx.path_graph(4)
    assert nx.predecessor(G, 0) == {0: [], 1: [0], 2: [1], 3: [2]}
    assert nx.predecessor(G, 0, 3) == [2]