import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_copy_name(self):
    G = nx.Graph()
    H = nx.relabel_nodes(G, {}, copy=True)
    assert H.graph == G.graph
    H = nx.relabel_nodes(G, {}, copy=False)
    assert H.graph == G.graph
    G.name = 'first'
    H = nx.relabel_nodes(G, {}, copy=True)
    assert H.graph == G.graph
    H = nx.relabel_nodes(G, {}, copy=False)
    assert H.graph == G.graph