import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_star(self):
    G = nx.star_graph(5)
    L = nx.line_graph(G)
    assert nx.is_isomorphic(L, nx.complete_graph(5))