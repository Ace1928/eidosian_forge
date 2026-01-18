import pytest
import networkx as nx
def test_predecessor_target(self):
    G = nx.path_graph(4)
    p = nx.predecessor(G, 0, 3)
    assert p == [2]
    p = nx.predecessor(G, 0, 3, cutoff=2)
    assert p == []
    p, s = nx.predecessor(G, 0, 3, return_seen=True)
    assert p == [2]
    assert s == 3
    p, s = nx.predecessor(G, 0, 3, cutoff=2, return_seen=True)
    assert p == []
    assert s == -1