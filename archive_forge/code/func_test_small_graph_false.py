import pytest
import networkx as nx
def test_small_graph_false():
    z = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')
    z = [6, 5, 4, 4, 2, 1, 1, 1]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')
    z = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')