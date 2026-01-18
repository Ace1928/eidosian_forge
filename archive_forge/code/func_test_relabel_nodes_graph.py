import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_graph(self):
    G = nx.Graph([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])
    mapping = {'A': 'aardvark', 'B': 'bear', 'C': 'cat', 'D': 'dog'}
    H = nx.relabel_nodes(G, mapping)
    assert nodes_equal(H.nodes(), ['aardvark', 'bear', 'cat', 'dog'])