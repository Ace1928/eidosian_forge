import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_planar_digraph(self):
    G = nx.DiGraph([(1, 2), (2, 3), (2, 4), (4, 1), (4, 2), (1, 4), (3, 2)])
    self.check_graph(G, is_planar=True)