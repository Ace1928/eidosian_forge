import pytest
import networkx as nx
def test_notarborescence1():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (5, 6)])
    assert nx.is_branching(G)
    assert not nx.is_arborescence(G)