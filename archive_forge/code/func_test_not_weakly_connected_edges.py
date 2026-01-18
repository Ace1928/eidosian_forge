import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_not_weakly_connected_edges():
    with pytest.raises(nx.NetworkXNoPath):
        G = nx.DiGraph()
        nx.add_path(G, [1, 2, 3])
        nx.add_path(G, [4, 5])
        list(nx.edge_disjoint_paths(G, 1, 5))