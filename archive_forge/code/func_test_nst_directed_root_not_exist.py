import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_directed_root_not_exist(self):
    G = nx.empty_graph(3, create_using=nx.MultiDiGraph)
    with pytest.raises(nx.NetworkXError):
        nx.number_of_spanning_trees(G, root=42)