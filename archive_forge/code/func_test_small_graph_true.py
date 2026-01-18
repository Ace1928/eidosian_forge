import pytest
import networkx as nx
def test_small_graph_true():
    z = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')
    z = [10, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')
    z = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')