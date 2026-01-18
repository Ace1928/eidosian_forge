import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_star_graph(self):
    assert is_isomorphic(nx.star_graph(''), nx.empty_graph(0))
    assert is_isomorphic(nx.star_graph([]), nx.empty_graph(0))
    assert is_isomorphic(nx.star_graph(0), nx.empty_graph(1))
    assert is_isomorphic(nx.star_graph(1), nx.path_graph(2))
    assert is_isomorphic(nx.star_graph(2), nx.path_graph(3))
    assert is_isomorphic(nx.star_graph(5), nx.complete_bipartite_graph(1, 5))
    s = nx.star_graph(10)
    assert sorted((d for n, d in s.degree())) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
    pytest.raises(nx.NetworkXError, nx.star_graph, 10, create_using=nx.DiGraph)
    ms = nx.star_graph(10, create_using=nx.MultiGraph)
    assert edges_equal(ms.edges(), s.edges())
    G = nx.star_graph('abc')
    assert len(G) == 3
    assert G.size() == 2
    G = nx.star_graph('abcb')
    assert len(G) == 3
    assert G.size() == 2
    G = nx.star_graph('abcb', create_using=nx.MultiGraph)
    assert len(G) == 3
    assert G.size() == 3
    G = nx.star_graph('abcdefg')
    assert len(G) == 7
    assert G.size() == 6