import math
from functools import partial
import pytest
import networkx as nx
def test_zero_degrees(self):
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    self.test(G, [(0, 1)], [(0, 1, 0)])