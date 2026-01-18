import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_callable_type(self):
    G = nx.path_graph(4)
    H = nx.relabel_nodes(G, str)
    assert nodes_equal(H.nodes, ['0', '1', '2', '3'])