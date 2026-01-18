import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_missing(self):
    G = nx.Graph([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])
    mapping = {0: 'aardvark'}
    H = nx.relabel_nodes(G, mapping, copy=True)
    assert nodes_equal(H.nodes, G.nodes)
    GG = G.copy()
    nx.relabel_nodes(G, mapping, copy=False)
    assert nodes_equal(G.nodes, GG.nodes)