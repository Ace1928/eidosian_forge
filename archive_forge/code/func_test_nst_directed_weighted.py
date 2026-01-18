import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_directed_weighted(self):
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 3, weight=1)
    G.add_edge(2, 3, weight=3)
    Nst = nx.number_of_spanning_trees(G, root=1, weight='weight')
    assert np.isclose(Nst, 8)
    Nst = nx.number_of_spanning_trees(G, root=2, weight='weight')
    assert np.isclose(Nst, 0)
    Nst = nx.number_of_spanning_trees(G, root=3, weight='weight')
    assert np.isclose(Nst, 0)