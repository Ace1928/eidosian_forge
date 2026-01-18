import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_path_graph(self):
    p = nx.path_graph(0)
    assert is_isomorphic(p, nx.null_graph())
    p = nx.path_graph(1)
    assert is_isomorphic(p, nx.empty_graph(1))
    p = nx.path_graph(10)
    assert nx.is_connected(p)
    assert sorted((d for n, d in p.degree())) == [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    assert p.order() - 1 == p.size()
    dp = nx.path_graph(3, create_using=nx.DiGraph)
    assert dp.has_edge(0, 1)
    assert not dp.has_edge(1, 0)
    mp = nx.path_graph(10, create_using=nx.MultiGraph)
    assert edges_equal(mp.edges(), p.edges())
    G = nx.path_graph('abc')
    assert len(G) == 3
    assert G.size() == 2
    G = nx.path_graph('abcb')
    assert len(G) == 3
    assert G.size() == 2
    g = nx.path_graph('abc', nx.DiGraph)
    assert len(g) == 3
    assert g.size() == 2
    assert g.is_directed()
    g = nx.path_graph('abcb', nx.DiGraph)
    assert len(g) == 3
    assert g.size() == 3
    G = nx.path_graph((1, 2, 3, 2, 4))
    assert G.has_edge(2, 4)