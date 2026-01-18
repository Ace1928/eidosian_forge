import pytest
import networkx as nx
def test_notarborescence2():
    G = nx.MultiDiGraph()
    nx.add_path(G, range(5))
    G.add_edge(6, 4)
    assert not nx.is_branching(G)
    assert not nx.is_arborescence(G)