import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_planar_with_selfloop(self):
    e = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (1, 2), (1, 3), (1, 5), (2, 5), (2, 4), (3, 4), (3, 5), (4, 5)]
    self.check_graph(nx.Graph(e), is_planar=True)