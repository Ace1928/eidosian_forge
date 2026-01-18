import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_planar_multigraph(self):
    G = nx.MultiGraph([(1, 2), (1, 2), (1, 2), (1, 2), (2, 3), (3, 1)])
    self.check_graph(G, is_planar=True)