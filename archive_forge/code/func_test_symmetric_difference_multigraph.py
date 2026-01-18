import os
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.utils import edges_equal
def test_symmetric_difference_multigraph():
    g = nx.MultiGraph()
    g.add_edge(0, 1, key=0)
    g.add_edge(0, 1, key=1)
    g.add_edge(0, 1, key=2)
    h = nx.MultiGraph()
    h.add_edge(0, 1, key=0)
    h.add_edge(0, 1, key=3)
    gh = nx.symmetric_difference(g, h)
    assert set(gh.nodes()) == set(g.nodes())
    assert set(gh.nodes()) == set(h.nodes())
    assert sorted(gh.edges()) == 3 * [(0, 1)]
    assert sorted((sorted(e) for e in gh.edges(keys=True))) == [[0, 1, 1], [0, 1, 2], [0, 1, 3]]