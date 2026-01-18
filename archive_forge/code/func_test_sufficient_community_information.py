import math
from functools import partial
import pytest
import networkx as nx
def test_sufficient_community_information(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    G.nodes[1]['community'] = 0
    G.nodes[2]['community'] = 0
    G.nodes[3]['community'] = 0
    G.nodes[4]['community'] = 0
    self.test(G, [(1, 4)], [(1, 4, 2 / self.delta)])