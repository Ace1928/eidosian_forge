import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def test_not_sortable_nodes(self):
    G = nx.Graph([(0, 'a')])
    treewidth_min_fill_in(G)