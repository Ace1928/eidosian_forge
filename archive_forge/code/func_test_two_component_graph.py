import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def test_two_component_graph(self):
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    treewidth, _ = treewidth_min_fill_in(G)
    assert treewidth == 0