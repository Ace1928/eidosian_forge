import pytest
import networkx as nx
def test_relaxed_caveman_graph():
    G = nx.relaxed_caveman_graph(4, 3, 0)
    assert len(G) == 12
    G = nx.relaxed_caveman_graph(4, 3, 1)
    assert len(G) == 12
    G = nx.relaxed_caveman_graph(4, 3, 0.5)
    assert len(G) == 12
    G = nx.relaxed_caveman_graph(4, 3, 0.5, seed=42)
    assert len(G) == 12