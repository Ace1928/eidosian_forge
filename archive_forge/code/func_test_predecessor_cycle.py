import pytest
import networkx as nx
def test_predecessor_cycle(self):
    G = nx.cycle_graph(4)
    pred = nx.predecessor(G, 0)
    assert pred[0] == []
    assert pred[1] == [0]
    assert pred[2] in [[1, 3], [3, 1]]
    assert pred[3] == [0]