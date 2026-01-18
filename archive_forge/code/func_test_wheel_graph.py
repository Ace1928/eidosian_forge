import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_wheel_graph(self):
    for n, G in [('', nx.null_graph()), (0, nx.null_graph()), (1, nx.empty_graph(1)), (2, nx.path_graph(2)), (3, nx.complete_graph(3)), (4, nx.complete_graph(4))]:
        g = nx.wheel_graph(n)
        assert is_isomorphic(g, G)
    g = nx.wheel_graph(10)
    assert sorted((d for n, d in g.degree())) == [3, 3, 3, 3, 3, 3, 3, 3, 3, 9]
    pytest.raises(nx.NetworkXError, nx.wheel_graph, 10, create_using=nx.DiGraph)
    mg = nx.wheel_graph(10, create_using=nx.MultiGraph())
    assert edges_equal(mg.edges(), g.edges())
    G = nx.wheel_graph('abc')
    assert len(G) == 3
    assert G.size() == 3
    G = nx.wheel_graph('abcb')
    assert len(G) == 3
    assert G.size() == 4
    G = nx.wheel_graph('abcb', nx.MultiGraph)
    assert len(G) == 3
    assert G.size() == 6