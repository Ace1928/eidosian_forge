from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_graph_with_reported_bug(self):
    G = nx.DiGraph()
    edges = [(0, 2), (0, 3), (1, 0), (1, 3), (2, 1), (2, 4), (3, 2), (3, 4), (4, 0), (4, 1), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3)]
    G.add_edges_from(edges)
    cc = sorted(nx.simple_cycles(G))
    assert len(cc) == 26
    rcc = sorted(nx.recursive_simple_cycles(G))
    assert len(cc) == len(rcc)
    for c in cc:
        assert any((self.is_cyclic_permutation(c, rc) for rc in rcc))
    for rc in rcc:
        assert any((self.is_cyclic_permutation(rc, c) for c in cc))