import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_function(self):
    G = nx.empty_graph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])

    def mapping(n):
        return ord(n)
    H = nx.relabel_nodes(G, mapping)
    assert nodes_equal(H.nodes(), [65, 66, 67, 68])