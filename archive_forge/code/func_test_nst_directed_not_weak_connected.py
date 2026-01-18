import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_directed_not_weak_connected(self):
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    assert np.isclose(nx.number_of_spanning_trees(G, root=1), 0)