import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_multidigraph(self):
    G = nx.MultiDiGraph([('a', 'b'), ('a', 'b')])
    mapping = {'a': 'aardvark', 'b': 'bear'}
    G = nx.relabel_nodes(G, mapping, copy=False)
    assert nodes_equal(G.nodes(), ['aardvark', 'bear'])
    assert edges_equal(G.edges(), [('aardvark', 'bear'), ('aardvark', 'bear')])