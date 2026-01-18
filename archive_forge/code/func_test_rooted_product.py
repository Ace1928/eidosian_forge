import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_rooted_product():
    G = nx.cycle_graph(5)
    H = nx.Graph()
    H.add_edges_from([('a', 'b'), ('b', 'c'), ('b', 'd')])
    R = nx.rooted_product(G, H, 'a')
    assert len(R) == len(G) * len(H)
    assert R.size() == G.size() + len(G) * H.size()