import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_selfloop(self):
    G = nx.DiGraph([(1, 1), (1, 2), (2, 3)])
    G = nx.relabel_nodes(G, {1: 'One', 2: 'Two', 3: 'Three'}, copy=False)
    assert nodes_equal(G.nodes(), ['One', 'Three', 'Two'])
    G = nx.MultiDiGraph([(1, 1), (1, 2), (2, 3)])
    G = nx.relabel_nodes(G, {1: 'One', 2: 'Two', 3: 'Three'}, copy=False)
    assert nodes_equal(G.nodes(), ['One', 'Three', 'Two'])
    G = nx.MultiDiGraph([(1, 1)])
    G = nx.relabel_nodes(G, {1: 0}, copy=False)
    assert nodes_equal(G.nodes(), [0])