from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_periodic_multigraph(self):
    G = nx.grid_2d_graph(4, 2, periodic=True)
    H = nx.grid_2d_graph(4, 2, periodic=True, create_using=nx.MultiGraph())
    assert list(G.edges()) == list(H.edges())