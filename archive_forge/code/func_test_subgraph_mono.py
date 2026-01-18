import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_subgraph_mono(self):
    g1 = nx.Graph()
    g2 = nx.Graph()
    g1.add_edges_from(self.g1edges)
    g2.add_edges_from([[1, 2], [2, 3], [3, 4]])
    gm = iso.GraphMatcher(g1, g2)
    assert gm.subgraph_is_monomorphic()